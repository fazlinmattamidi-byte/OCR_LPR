'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  ArrowLeft,
  SwitchCamera,
  Zap,
  ZapOff,
  AlertOctagon,
  AlertTriangle,
  Pause,
  Play,
  Settings as SettingsIcon,
  ChevronDown,
  ChevronUp,
  LayoutGrid,
  Search,
  Camera,
  Car,
} from 'lucide-react';
import Link from 'next/link';
import { PlateTracker, ActiveTrack } from '@/lib/anpr/tracker';
import {
  detectMalaysianPlates,
  validateDetector,
  getActiveDetectorProvider,
  ActiveExecutionProvider,
} from '@/lib/anpr/yoloDetector';
import {
  generateAdaptiveCrops,
  createInnerPlateTextCrop,
  cropCanvasRegionFast,
  prioritiseTracks,
  releaseCanvasMemory,
} from '@/lib/anpr/imageProcessor';
import { globalBestFrameSelector } from '@/lib/anpr/bestFrameSelector';
import { recognizePlateFromCanvas } from '@/lib/anpr/ocrEngine';
import { addOcrVoteToTrack, evaluateConsensus } from '@/lib/anpr/consensus';
import { playAlertSound, triggerVibration } from '@/lib/utils/audio';
import { VehicleCase, ScannerSettings } from '@/lib/db/types';
import { INITIAL_SETTINGS } from '@/lib/db/settingsDefaults';
import { ModelStatusBanner } from '@/components/scanner/ModelStatusBanner';
import {
  initializeANPRRuntime,
  getANPRRuntimeState,
  getLatestBenchmarkResult,
  getRuntimeErrorMessage,
  ANPRRuntimeState,
  AdmissionBenchmarkResult,
} from '@/lib/anpr/runtimeManager';
import { getActivePpOcrProvider, isPpOcrReady, ActiveOcrProvider } from '@/lib/anpr/ppOcrEngine';
import { validateMalaysianPattern } from '@/lib/anpr/patterns';

interface MatchEntry {
  type: 'EXACT' | 'POSSIBLE';
  plate: string;
  trackId: string;
  vehicle: VehicleCase | null;
  possibleMatches: VehicleCase[];
  confidence: number;
  scanId?: string;
  timestamp: number;
  dismissed: boolean;
}

function getTrackColor(track: ActiveTrack): string {
  if (track.matchType === 'EXACT') return '#ef4444';   // red
  if (track.matchType === 'POSSIBLE') return '#f59e0b'; // amber
  if (track.matchType === 'NONE') return '#00d8f6';    // cyan
  if (track.ocrState === 'COOLDOWN') return '#6b7280'; // grey
  return '#00d8f6';  // cyan — default / reading
}

function getTrackStatusLabel(track: ActiveTrack): string {
  if (track.matchType === 'EXACT') return 'MATCH';
  if (track.matchType === 'POSSIBLE') return 'POSSIBLE';
  if (track.matchType === 'NONE') return 'NO CASE';
  switch (track.ocrState) {
    case 'DETECTED': return 'DETECTED';
    case 'COLLECTING': return 'COLLECTING';
    case 'OCR_RUNNING': return 'READING…';
    case 'CONSENSUS_BUILDING': return 'ANALYSING';
    case 'DB_CHECKING': return 'CHECKING…';
    case 'COOLDOWN': return 'COOLDOWN';
    default: return 'SCANNING';
  }
}

function getTrackPlateText(track: ActiveTrack): string {
  if (track.stabilizedPlate) return track.stabilizedPlate;
  if (!track.votes || track.votes.size === 0) return '';

  let topPlate = '';
  let topCount = 0;
  let topTotalConfidence = 0;
  track.votes.forEach((data, plateStr) => {
    if (data.count > topCount || (data.count === topCount && data.totalConfidence > topTotalConfidence)) {
      topPlate = plateStr;
      topCount = data.count;
      topTotalConfidence = data.totalConfidence;
    }
  });

  return topPlate;
}

function isRuntimeScanningReady(state: ANPRRuntimeState): boolean {
  return state === 'READY_WEBGPU' || state === 'READY_WASM' || state === 'DEGRADED_PERFORMANCE';
}

const DETECTION_TARGET_INTERVAL_MS = 90;
const DETECTION_BUSY_INTERVAL_MS = 125;
const DETECTION_MIN_DELAY_MS = 16;
const DETECTOR_VALIDATION_INTERVAL_MS = 2000;
const CROP_SAMPLE_FAST_MS = 90;
const CROP_SAMPLE_NORMAL_MS = 160;
const OCR_FIRST_READ_RETRY_MS = 80;
const OCR_REPEAT_READ_RETRY_MS = 180;
const OCR_MAX_CONCURRENCY = 4;
const OVERLAY_ANGLE_SAMPLE_MS = 140;
const MAX_OVERLAY_TILT_RAD = 0.35;
const SCANNER_MAINTENANCE_INTERVAL_MS = 1000;
const COOLDOWN_MAP_MAX_ENTRIES = 120;
const MATCH_ALERT_LIMIT = 6;
const TRACK_RESULT_HOLD_MS = 900;

function getExpectedMinPlateChars(crop: HTMLCanvasElement): number {
  const aspect = crop.width / Math.max(1, crop.height);
  if (aspect >= 3.0) return 5;
  if (aspect >= 2.3) return 4;
  return 3;
}

function clampNumber(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function pruneCooldownMap(cooldowns: Map<string, number>, cooldownMs: number, now: number): void {
  const maxAgeMs = Math.max(cooldownMs * 2, 30000);
  for (const [plate, timestamp] of cooldowns.entries()) {
    if (now - timestamp > maxAgeMs) {
      cooldowns.delete(plate);
    }
  }

  if (cooldowns.size <= COOLDOWN_MAP_MAX_ENTRIES) return;

  const entries = Array.from(cooldowns.entries()).sort((a, b) => b[1] - a[1]);
  cooldowns.clear();
  entries.slice(0, COOLDOWN_MAP_MAX_ENTRIES).forEach(([plate, timestamp]) => {
    cooldowns.set(plate, timestamp);
  });
}

function enqueueMatchEntry(queue: MatchEntry[], entry: MatchEntry): MatchEntry[] {
  return [...queue.filter(m => m.plate !== entry.plate), entry].slice(-MATCH_ALERT_LIMIT);
}

function resetTrackForNextPlate(track: ActiveTrack): void {
  track.ocrState = 'COLLECTING';
  track.ocrRunning = false;
  track.ocrJobQueued = false;
  track.cooldownActive = false;
  track.votes.clear();
  track.cropSamples = [];

  delete track.cooldownStartedAt;
  delete track.lastCropSampledAt;
  delete track.lastOcrAttemptAt;
  delete track.lastOcrCompletedAt;
  delete track.lastSearchedAt;
  delete track.stabilizedPlate;
  delete track.stabilizedConfidence;
  delete track.matchType;
  delete track.matchedVehicle;
  delete track.possibleMatchVehicles;
  delete track.scanEventId;
}

function drawRoundedRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number
): void {
  if (typeof ctx.roundRect === 'function') {
    ctx.roundRect(x, y, width, height, radius);
    return;
  }

  const r = Math.min(radius, Math.abs(width) / 2, Math.abs(height) / 2);
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + width - r, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + r);
  ctx.lineTo(x + width, y + height - r);
  ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  ctx.lineTo(x + r, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
}

function countPlateChars(text: string): { letters: number; digits: number } {
  return {
    letters: (text.match(/[A-Z]/g) || []).length,
    digits: (text.match(/[0-9]/g) || []).length,
  };
}

function isPlausiblePlateCandidate(
  text: string,
  expectedMinChars: number,
  patternScore: number
): boolean {
  if (!text || text.length < 3) return false;

  const { letters, digits } = countPlateChars(text);
  if (letters === 0 || digits === 0) return false;

  const hasEnoughLength = text.length >= expectedMinChars || (text.length >= 3 && patternScore >= 0.85);
  if (!hasEnoughLength) return false;

  if (text.length >= 5 && digits < 2) return false;
  if (/^([A-Z0-9]{2,4})\1$/.test(text)) return false;

  return true;
}

function canCommitFinalPlateOutcome(text: string): boolean {
  const pattern = validateMalaysianPattern(text);
  const { letters, digits } = countPlateChars(text);

  if (!pattern.isValid || pattern.score < 0.55) return false;
  if (letters === 0 || digits === 0) return false;
  if (text.length >= 5 && digits < 2) return false;

  return true;
}

function estimatePlateOverlayAngle(
  sourceCanvas: HTMLCanvasElement,
  bbox: { x: number; y: number; width: number; height: number },
  previousAngle = 0
): number {
  const sourceWidth = sourceCanvas.width;
  const sourceHeight = sourceCanvas.height;
  const x = clampNumber(Math.round(bbox.x), 0, Math.max(0, sourceWidth - 1));
  const y = clampNumber(Math.round(bbox.y), 0, Math.max(0, sourceHeight - 1));
  const width = clampNumber(Math.round(bbox.width), 1, sourceWidth - x);
  const height = clampNumber(Math.round(bbox.height), 1, sourceHeight - y);

  if (width < 30 || height < 10) return previousAngle;

  const ctx = sourceCanvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return previousAngle;

  try {
    const image = ctx.getImageData(x, y, width, height);
    const { data } = image;
    const step = Math.max(1, Math.floor(Math.max(width, height) / 90));
    let weightSum = 0;
    let meanX = 0;
    let meanY = 0;

    const lumaAt = (px: number, py: number) => {
      const idx = (py * width + px) * 4;
      return data[idx] * 0.299 + data[idx + 1] * 0.587 + data[idx + 2] * 0.114;
    };

    for (let py = step; py < height - step; py += step) {
      for (let px = step; px < width - step; px += step) {
        const gx = Math.abs(lumaAt(px + step, py) - lumaAt(px - step, py));
        const gy = Math.abs(lumaAt(px, py + step) - lumaAt(px, py - step));
        const edge = gx + gy;
        if (edge < 28) continue;
        weightSum += edge;
        meanX += px * edge;
        meanY += py * edge;
      }
    }

    if (weightSum < 1) return previousAngle;

    meanX /= weightSum;
    meanY /= weightSum;

    let covXX = 0;
    let covXY = 0;
    let covYY = 0;

    for (let py = step; py < height - step; py += step) {
      for (let px = step; px < width - step; px += step) {
        const gx = Math.abs(lumaAt(px + step, py) - lumaAt(px - step, py));
        const gy = Math.abs(lumaAt(px, py + step) - lumaAt(px, py - step));
        const edge = gx + gy;
        if (edge < 28) continue;
        const dx = px - meanX;
        const dy = py - meanY;
        covXX += dx * dx * edge;
        covXY += dx * dy * edge;
        covYY += dy * dy * edge;
      }
    }

    let angle = 0.5 * Math.atan2(2 * covXY, covXX - covYY);
    if (angle > Math.PI / 2) angle -= Math.PI;
    if (angle < -Math.PI / 2) angle += Math.PI;
    if (Math.abs(angle) > MAX_OVERLAY_TILT_RAD) angle = previousAngle;

    return previousAngle * 0.60 + angle * 0.40;
  } catch {
    return previousAngle;
  }
}

function getCameraPreflightError(): string | null {
  if (typeof window === 'undefined') return null;

  if (!window.isSecureContext) {
    return `Camera access is blocked on this address (${window.location.host}). Open the scanner over HTTPS, or use localhost on the same desktop/laptop. iPhone Safari and Android browsers require HTTPS for camera access.`;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    return 'This browser does not expose camera access. Try a current mobile browser and allow camera permission.';
  }

  return null;
}

function getResolutionConstraints(
  preferredResolution: ScannerSettings['preferredResolution']
): Pick<MediaTrackConstraints, 'width' | 'height'> {
  if (preferredResolution === '1080p') {
    return { width: { ideal: 1920 }, height: { ideal: 1080 } };
  }

  if (preferredResolution === '480p') {
    return { width: { ideal: 854 }, height: { ideal: 480 } };
  }

  return { width: { ideal: 1280 }, height: { ideal: 720 } };
}

function isSavedDeviceId(preferredCamera?: string): boolean {
  return !!preferredCamera && !['environment', 'user', 'default', 'any'].includes(preferredCamera);
}

function buildCameraConstraintCandidates(
  settings: ScannerSettings,
  requestedDeviceId?: string
): Array<MediaTrackConstraints | boolean> {
  const resolution = getResolutionConstraints(settings.preferredResolution);
  const candidates: Array<MediaTrackConstraints | boolean> = [];
  const seen = new Set<string>();
  const add = (video: MediaTrackConstraints | boolean) => {
    const key = typeof video === 'boolean' ? String(video) : JSON.stringify(video);
    if (!seen.has(key)) {
      seen.add(key);
      candidates.push(video);
    }
  };

  if (requestedDeviceId) {
    add({ ...resolution, deviceId: { exact: requestedDeviceId } });
    add({ ...resolution, deviceId: { ideal: requestedDeviceId } });
  }

  if (!requestedDeviceId && isSavedDeviceId(settings.preferredCamera)) {
    add({ ...resolution, deviceId: { exact: settings.preferredCamera } });
    add({ ...resolution, deviceId: { ideal: settings.preferredCamera } });
  }

  const preferredFacing = settings.preferredCamera === 'user' ? 'user' : 'environment';
  const fallbackFacing = preferredFacing === 'environment' ? 'user' : 'environment';

  add({ ...resolution, facingMode: { ideal: preferredFacing } });
  add({ ...resolution, facingMode: { ideal: fallbackFacing } });
  add({ ...resolution });
  add(true);

  return candidates;
}

async function enumerateVideoDevices(): Promise<MediaDeviceInfo[]> {
  try {
    const allDevices = await navigator.mediaDevices.enumerateDevices();
    return allDevices.filter(d => d.kind === 'videoinput');
  } catch {
    return [];
  }
}

async function openCompatibleCamera(
  settings: ScannerSettings,
  requestedDeviceId?: string
): Promise<MediaStream> {
  const candidates = buildCameraConstraintCandidates(settings, requestedDeviceId);
  let lastError: any = null;

  for (const video of candidates) {
    try {
      return await navigator.mediaDevices.getUserMedia({ audio: false, video });
    } catch (err) {
      lastError = err;
      console.warn('[Scanner] Camera constraint failed, trying fallback:', err);
    }
  }

  throw lastError || new Error('No compatible camera constraints found.');
}

async function attachCameraStream(video: HTMLVideoElement, stream: MediaStream): Promise<void> {
  video.srcObject = stream;
  video.muted = true;
  video.playsInline = true;

  if (video.readyState < HTMLMediaElement.HAVE_METADATA || video.videoWidth === 0) {
    await new Promise<void>((resolve, reject) => {
      const timeout = window.setTimeout(() => resolve(), 2500);
      const cleanup = () => {
        window.clearTimeout(timeout);
        video.removeEventListener('loadedmetadata', onReady);
        video.removeEventListener('canplay', onReady);
        video.removeEventListener('error', onError);
      };
      const onReady = () => {
        cleanup();
        resolve();
      };
      const onError = () => {
        cleanup();
        reject(new Error('Camera stream could not attach to video preview.'));
      };

      video.addEventListener('loadedmetadata', onReady, { once: true });
      video.addEventListener('canplay', onReady, { once: true });
      video.addEventListener('error', onError, { once: true });
    });
  }

  await video.play();
}

async function applyCameraQualityOptimizations(track: MediaStreamTrack): Promise<void> {
  if (typeof track.getCapabilities !== 'function' || typeof track.applyConstraints !== 'function') return;

  try {
    const caps = track.getCapabilities() as any;
    const advanced: any[] = [];

    if (Array.isArray(caps.focusMode) && caps.focusMode.includes('continuous')) {
      advanced.push({ focusMode: 'continuous' });
    }
    if (Array.isArray(caps.exposureMode) && caps.exposureMode.includes('continuous')) {
      advanced.push({ exposureMode: 'continuous' });
    }
    if (Array.isArray(caps.whiteBalanceMode) && caps.whiteBalanceMode.includes('continuous')) {
      advanced.push({ whiteBalanceMode: 'continuous' });
    }

    if (advanced.length > 0) {
      await track.applyConstraints({ advanced } as any);
    }
  } catch (err) {
    console.warn('[Scanner] Camera quality constraints unavailable:', err);
  }
}

function getCameraErrorMessage(err: any): string {
  if (err?.name === 'NotAllowedError') {
    return 'Camera permission denied. Allow camera access in browser settings and retry.';
  }
  if (err?.name === 'NotFoundError') {
    return 'No camera device found. Connect a webcam or enable the built-in camera, then retry.';
  }
  if (err?.name === 'NotReadableError') {
    return 'Camera is already in use by another application.';
  }
  if (err?.name === 'OverconstrainedError') {
    return 'Requested camera settings are not supported by this device. Retrying with a different camera or lower resolution may help.';
  }
  if (err?.name === 'SecurityError') {
    return 'Camera access requires HTTPS or localhost.';
  }

  return err?.message || 'Unable to access camera.';
}

export default function ScannerPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const processingCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const trackerRef = useRef<PlateTracker>(new PlateTracker(20, 8));
  const streamRef = useRef<MediaStream | null>(null);
  const runtimeStateRef = useRef<ANPRRuntimeState>('UNINITIALIZED');

  // Camera state
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [torchOn, setTorchOn] = useState(false);
  const [torchSupported, setTorchSupported] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [cameraReady, setCameraReady] = useState(false);

  // Scanner control
  const isPausedRef = useRef<boolean>(false);
  const [isPaused, setIsPaused] = useState(false);

  // ANPR Production Runtime State Machine
  const [runtimeState, setRuntimeState] = useState<ANPRRuntimeState>('UNINITIALIZED');
  const [benchmarkResult, setBenchmarkResult] = useState<AdmissionBenchmarkResult | null>(null);
  const [runtimeErrorMessage, setRuntimeErrorMessage] = useState<string | null>(null);

  // Detector engine state
  const [activeEngine, setActiveEngine] = useState<'LOCAL_ONNX' | 'CV_HEURISTIC'>('LOCAL_ONNX');
  const [detectorProvider, setDetectorProvider] = useState<ActiveExecutionProvider>('NONE');
  const [ocrProvider, setOcrProvider] = useState<ActiveOcrProvider>('NONE');
  const [avgConfidence, setAvgConfidence] = useState<number>(0.85);

  // Performance metrics
  const [camFps, setCamFps] = useState(0);
  const [detFps, setDetFps] = useState(0);
  const [platesVisible, setPlatesVisible] = useState(0);
  const [activeTracksCount, setActiveTracksCount] = useState(0);

  // Active tracks for results tray
  const [tracksList, setTracksList] = useState<ActiveTrack[]>([]);
  const [trayExpanded, setTrayExpanded] = useState(true);

  // Match queue — all active matches, shown simultaneously
  const [matchQueue, setMatchQueue] = useState<MatchEntry[]>([]);
  const [viewingMatch, setViewingMatch] = useState<MatchEntry | null>(null);

  // Settings
  const settingsRef = useRef<ScannerSettings>({ ...INITIAL_SETTINGS, debugMode: false });
  const [isDebugMode, setIsDebugMode] = useState<boolean>(false);

  const camFrameCount = useRef(0);
  const detFrameCount = useRef(0);
  const lastFpsTs = useRef(Date.now());
  const cooldownMap = useRef<Map<string, number>>(new Map());
  const activeOcrCount = useRef(0);
  const lastMaintenanceTs = useRef(0);

  const resetLiveScanUi = useCallback(() => {
    trackerRef.current.clear();
    globalBestFrameSelector.resetAll();
    camFrameCount.current = 0;
    detFrameCount.current = 0;
    activeOcrCount.current = 0;
    lastMaintenanceTs.current = 0;
    setTracksList([]);
    setPlatesVisible(0);
    setActiveTracksCount(0);
    setCamFps(0);
    setDetFps(0);
  }, []);

  const syncRuntimeStatus = useCallback(() => {
    const nextState = getANPRRuntimeState();
    runtimeStateRef.current = nextState;
    setRuntimeState(prev => prev === nextState ? prev : nextState);

    const nextBenchmark = getLatestBenchmarkResult();
    setBenchmarkResult(prev => prev === nextBenchmark ? prev : nextBenchmark);
    setRuntimeErrorMessage(getRuntimeErrorMessage());
    setDetectorProvider(getActiveDetectorProvider());
    setOcrProvider(getActivePpOcrProvider());
  }, []);

  const startRuntimeInit = useCallback(async () => {
    runtimeStateRef.current = 'LOADING_MODELS';
    setRuntimeState('LOADING_MODELS');
    setRuntimeErrorMessage(null);

    const progressTimer = window.setInterval(syncRuntimeStatus, 500);
    try {
      const res = await initializeANPRRuntime();
      runtimeStateRef.current = res.state;
      setRuntimeState(res.state);
      if (res.benchmark) setBenchmarkResult(res.benchmark);
    } finally {
      window.clearInterval(progressTimer);
      syncRuntimeStatus();
    }
  }, [syncRuntimeStatus]);

  // ─── 1. Load Settings & Initialize Runtime ───────────────────────────
  useEffect(() => {
    fetch('/api/settings')
      .then(r => r.json())
      .then(data => {
        if (data.success && data.settings) {
          settingsRef.current = { ...settingsRef.current, ...data.settings };
          setIsDebugMode(!!data.settings.debugMode);
          trackerRef.current.setLostTrackTimeout(data.settings.lostTrackTimeout ?? 20);
          trackerRef.current.setMaxActiveTracks(data.settings.maxTracks ?? INITIAL_SETTINGS.maxTracks);
        }
      })
      .catch(() => {});

    startRuntimeInit();
  }, [startRuntimeInit]);

  useEffect(() => {
    const id = window.setInterval(syncRuntimeStatus, 1000);
    return () => window.clearInterval(id);
  }, [syncRuntimeStatus]);

  // ─── 2. Initialise Camera ────────────────────────────────────────────────
  const initCamera = useCallback(async (deviceId?: string) => {
    setCameraError(null);
    setCameraReady(false);

    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
        streamRef.current = null;
      }
      resetLiveScanUi();

      const preflightError = getCameraPreflightError();
      if (preflightError) {
        setTorchOn(false);
        setTorchSupported(false);
        setCameraError(preflightError);
        return;
      }

      const s = settingsRef.current;
      const stream = await openCompatibleCamera(s, deviceId);
      streamRef.current = stream;

      if (videoRef.current) {
        await attachCameraStream(videoRef.current, stream);
      }

      const videoDevs = await enumerateVideoDevices();
      setDevices(videoDevs);

      const track = stream.getVideoTracks()[0];
      if (track) {
        await applyCameraQualityOptimizations(track);

        const trackSettings = typeof track.getSettings === 'function' ? track.getSettings() : {};
        if (trackSettings.deviceId) {
          setSelectedDeviceId(trackSettings.deviceId);
        } else if (deviceId) {
          setSelectedDeviceId(deviceId);
        }

        const caps = typeof track.getCapabilities === 'function' ? track.getCapabilities() as any : {};
        setTorchSupported(!!caps?.torch);
      }

      setCameraReady(true);
    } catch (err: any) {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
        streamRef.current = null;
      }
      resetLiveScanUi();
      setTorchOn(false);
      setTorchSupported(false);
      setCameraError(getCameraErrorMessage(err));
    }
  }, [resetLiveScanUi]);

  useEffect(() => {
    initCamera();
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
      }
    };
  }, [initCamera]);

  // ─── 3. Torch Toggle ────────────────────────────────────────────────────
  const toggleTorch = async () => {
    const track = streamRef.current?.getVideoTracks()[0];
    if (!track) return;
    try {
      await track.applyConstraints({ advanced: [{ torch: !torchOn } as any] });
      setTorchOn(v => !v);
    } catch {}
  };

  // ─── 4. Camera Switch ────────────────────────────────────────────────────
  const handleSwitchCamera = () => {
    if (devices.length < 2) return;
    const idx = devices.findIndex(d => d.deviceId === selectedDeviceId);
    const next = devices[(idx + 1) % devices.length];
    setSelectedDeviceId(next.deviceId);
    initCamera(next.deviceId);
  };

  // ─── 5. Pause / Resume ──────────────────────────────────────────────────
  const togglePause = () => {
    isPausedRef.current = !isPausedRef.current;
    setIsPaused(isPausedRef.current);
  };

  // ─── 6. Per-Track Database Match ─────────────────────────────────────────
  const runDatabaseMatch = useCallback(async (
    track: ActiveTrack,
    plate: string,
    confidence: number,
    options: { commitNoCase?: boolean } = {}
  ) => {
    const commitNoCase = options.commitNoCase ?? true;
    const cooldownMs = settingsRef.current.duplicateCooldown * 1000;
    const now = Date.now();
    const lastSearch = cooldownMap.current.get(plate) ?? 0;
    const trackSearchThrottleMs = commitNoCase ? 0 : 1000;

    if (commitNoCase && now - lastSearch < cooldownMs) {
      track.ocrState = 'COOLDOWN';
      track.cooldownActive = true;
      track.cooldownStartedAt = now;
      return;
    }
    if (trackSearchThrottleMs > 0 && track.lastSearchedAt && now - track.lastSearchedAt < trackSearchThrottleMs) return;
    track.lastSearchedAt = now;

    track.ocrState = 'DB_CHECKING';

    try {
      const searchRes = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ plateNumber: plate, source: 'CAMERA', confidence }),
      }).then(r => r.json());

      if (!searchRes.success) {
        track.ocrState = 'CONSENSUS_BUILDING';
        return;
      }

      if (!commitNoCase && searchRes.matchType !== 'EXACT' && searchRes.matchType !== 'POSSIBLE') {
        track.ocrState = 'CONSENSUS_BUILDING';
        return;
      }

      cooldownMap.current.set(plate, now);
      const resolvedMatchType = searchRes.matchType === 'EXACT'
        ? 'EXACT'
        : searchRes.matchType === 'POSSIBLE'
          ? 'POSSIBLE'
          : 'NONE';

      const scanRes = await fetch('/api/scans', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          detectedPlate: plate,
          normalizedPlate: plate,
          confidence,
          matchType: resolvedMatchType,
          matchedVehicleId: searchRes.matchedVehicle?.id ?? undefined,
          source: 'CAMERA',
          trackId: track.trackId,
          frameCount: track.framesSeen,
          firstSeenAt: new Date(Date.now() - track.framesSeen * 33).toISOString(),
        }),
      }).then(r => r.json());

      track.matchType = resolvedMatchType;
      track.matchedVehicle = searchRes.matchedVehicle ?? undefined;
      track.possibleMatchVehicles = searchRes.possibleMatches ?? [];
      track.ocrState = track.matchType === 'EXACT' ? 'MATCHED' : track.matchType === 'POSSIBLE' ? 'POSSIBLE MATCH' : 'NO CASE';

      if (resolvedMatchType === 'EXACT') {
        if (settingsRef.current.soundEnabled) playAlertSound('EXACT_MATCH');
        if (settingsRef.current.vibrationEnabled) triggerVibration([200, 100, 200, 100]);

        const entry: MatchEntry = {
          type: 'EXACT',
          plate,
          trackId: track.trackId,
          vehicle: searchRes.matchedVehicle,
          possibleMatches: [],
          confidence,
          scanId: scanRes.scanEvent?.id,
          timestamp: now,
          dismissed: false,
        };
        setMatchQueue(q => enqueueMatchEntry(q, entry));

      } else if (resolvedMatchType === 'POSSIBLE') {
        if (settingsRef.current.soundEnabled) playAlertSound('POSSIBLE_MATCH');
        const entry: MatchEntry = {
          type: 'POSSIBLE',
          plate,
          trackId: track.trackId,
          vehicle: null,
          possibleMatches: searchRes.possibleMatches ?? [],
          confidence,
          scanId: scanRes.scanEvent?.id,
          timestamp: now,
          dismissed: false,
        };
        setMatchQueue(q => enqueueMatchEntry(q, entry));
      }

      track.ocrState = 'COOLDOWN';
      track.cooldownActive = true;
      track.cooldownStartedAt = Date.now();
      track.scanEventId = scanRes.scanEvent?.id;
    } catch (err) {
      console.warn('[Scanner] DB match error:', err);
      track.ocrState = 'COLLECTING';
    }
  }, []);

  // ─── 7. Main ANPR Pipeline Loop ──────────────────────────────────────────
  useEffect(() => {
    if (!cameraReady) return;

    let animId: number;
    let detectionTimeout: ReturnType<typeof setTimeout> | undefined;
    let stopped = false;
    let detTs = Date.now();
    let detCount = 0;
    let lastVideoTime = -1;
    let lastDetectorValidationAt = 0;

    const scheduleDetection = (delay = 100) => {
      if (stopped) return;
      detectionTimeout = setTimeout(runDetection, delay);
    };

    const runDetection = async () => {
      if (stopped) return;

      if (!videoRef.current) {
        scheduleDetection(150);
        return;
      }

      if (isPausedRef.current || !isRuntimeScanningReady(runtimeStateRef.current)) {
        scheduleDetection(250);
        return;
      }

      const video = videoRef.current;

      if (video.readyState < 2 || video.videoWidth === 0) {
        scheduleDetection(150);
        return;
      }

      if (!processingCanvasRef.current) {
        processingCanvasRef.current = document.createElement('canvas');
      }

      const processingCanvas = processingCanvasRef.current;
      if (processingCanvas.width !== video.videoWidth || processingCanvas.height !== video.videoHeight) {
        processingCanvas.width = video.videoWidth;
        processingCanvas.height = video.videoHeight;
      }

      const processingCtx = processingCanvas.getContext('2d', { willReadFrequently: true });
      if (!processingCtx) {
        scheduleDetection(150);
        return;
      }

      const detectionStartedAt = performance.now();
      processingCtx.drawImage(video, 0, 0, processingCanvas.width, processingCanvas.height);

      const s = settingsRef.current;

      // ── Step 1: Malaysian Plate Detector ──
      let detectedPlates: Awaited<ReturnType<typeof detectMalaysianPlates>> = [];
      try {
        detectedPlates = await detectMalaysianPlates(processingCanvas, {
          minConfidence: s.detectionThreshold,
          enginePreference: s.detectorEngine,
          developerMode: s.debugMode,
        });
      } catch (err) {
        console.warn('[Scanner] Detector frame failed:', err);
      }

      if (detectedPlates.length > 0) {
        setActiveEngine(detectedPlates[0].sourceEngine);
        const avgConf = detectedPlates.reduce((sum, p) => sum + p.confidence, 0) / detectedPlates.length;
        setAvgConfidence(avgConf);
      } else {
        const now = Date.now();
        if (now - lastDetectorValidationAt > DETECTOR_VALIDATION_INTERVAL_MS) {
          lastDetectorValidationAt = now;
          const val = await validateDetector();
          if (val.valid) setActiveEngine('LOCAL_ONNX');
        }
      }

      const bboxList = detectedPlates.map(p => ({
        x: p.bbox.x,
        y: p.bbox.y,
        width: p.bbox.width,
        height: p.bbox.height,
        confidence: p.confidence,
      }));

      // ── Step 2: Multi-Object ByteTrack ──
      const allTracks = trackerRef.current.updateTracks(bboxList);
      const confirmedTracks = trackerRef.current.getActiveTracks(true); // Only confirmed tracks
      const displayTracks = s.debugMode ? allTracks : confirmedTracks;
      const loopNow = Date.now();

      confirmedTracks.forEach(track => {
        if (track.cooldownActive && !track.cooldownStartedAt) {
          track.cooldownStartedAt = loopNow;
        }

        if (
          track.cooldownActive &&
          track.cooldownStartedAt &&
          loopNow - track.cooldownStartedAt >= TRACK_RESULT_HOLD_MS
        ) {
          resetTrackForNextPlate(track);
          globalBestFrameSelector.clearTrack(track.trackNumber);
        }
      });

      if (loopNow - lastMaintenanceTs.current >= SCANNER_MAINTENANCE_INTERVAL_MS) {
        const activeTrackNumbers = new Set(allTracks.map(track => track.trackNumber));
        globalBestFrameSelector.clearExcept(activeTrackNumbers);
        globalBestFrameSelector.pruneStale(undefined, activeTrackNumbers, loopNow);
        pruneCooldownMap(cooldownMap.current, s.duplicateCooldown * 1000, loopNow);
        lastMaintenanceTs.current = loopNow;
      }

      setPlatesVisible(bboxList.length);
      setActiveTracksCount(confirmedTracks.length); // Update metric to show confirmed count
      setTracksList([...displayTracks]); // UI List

      // ── Step 3: Best Frame Selection (Only on Confirmed Tracks) ──
      const cropSampleNow = loopNow;
      confirmedTracks.forEach(track => {
        if (!track.lastOverlayAngleAt || cropSampleNow - track.lastOverlayAngleAt >= OVERLAY_ANGLE_SAMPLE_MS) {
          track.overlayAngle = estimatePlateOverlayAngle(processingCanvas, track.bbox, track.overlayAngle ?? 0);
          track.lastOverlayAngleAt = cropSampleNow;
        }

        if (track.ocrState === 'COOLDOWN' || track.ocrState === 'MATCHED') return;
        const existingCrop = globalBestFrameSelector.getBestCrop(track.trackNumber);
        const sampleInterval = track.bbox.confidence >= 0.70 ? CROP_SAMPLE_FAST_MS : CROP_SAMPLE_NORMAL_MS;
        if (existingCrop && track.lastCropSampledAt && cropSampleNow - track.lastCropSampledAt < sampleInterval) {
          return;
        }

        const cropCanvas = cropCanvasRegionFast(processingCanvas, track.bbox);
        globalBestFrameSelector.addCropCandidate(track.trackNumber, cropCanvas, track.bbox);
        track.lastCropSampledAt = cropSampleNow;
      });

      // ── Step 4: OCR Priority Queue (Async Decoupled) ──
      processOcrQueue(confirmedTracks, processingCanvas, s);

      detCount++;
      const now = Date.now();
      if (now - detTs >= 1000) {
        setDetFps(detCount);
        detCount = 0;
        detTs = now;
      }

      const targetInterval = activeOcrCount.current > 0 ? DETECTION_BUSY_INTERVAL_MS : DETECTION_TARGET_INTERVAL_MS;
      const nextDelay = Math.max(DETECTION_MIN_DELAY_MS, Math.round(targetInterval - (performance.now() - detectionStartedAt)));
      scheduleDetection(nextDelay);
    };

    const processOcrQueue = async (confirmedTracks: ActiveTrack[], canvas: HTMLCanvasElement, s: ScannerSettings) => {
      if (!isPpOcrReady()) {
        confirmedTracks.forEach(track => {
          if (!track.cooldownActive && track.ocrState !== 'MATCHED') {
            track.ocrState = 'COLLECTING';
          }
        });
        return;
      }

      const maxOcrConcurrency = Math.min(
        OCR_MAX_CONCURRENCY,
        Math.max(1, s.maxOcrConcurrency || INITIAL_SETTINGS.maxOcrConcurrency)
      );
      const priorityIds = prioritiseTracks(
        confirmedTracks.map(t => ({
          trackId: t.trackId,
          bbox: t.bbox,
          framesSeen: t.framesSeen,
          ocrState: t.ocrState,
          lastOcrAttemptAt: t.lastOcrAttemptAt,
          voteCount: Array.from(t.votes.values()).reduce((sum, vote) => sum + vote.count, 0),
        })),
        canvas.width,
        canvas.height,
        maxOcrConcurrency
      );

      for (const trackId of priorityIds) {
        const track = trackerRef.current.getTrack(trackId);
        if (!track || !track.isConfirmed || track.ocrRunning || track.cooldownActive) continue;
        if (activeOcrCount.current >= maxOcrConcurrency) break;

        const now = Date.now();
        const voteCount = Array.from(track.votes.values()).reduce((sum, vote) => sum + vote.count, 0);
        const retryDelay = voteCount === 0 ? OCR_FIRST_READ_RETRY_MS : OCR_REPEAT_READ_RETRY_MS;
        if (track.lastOcrAttemptAt && now - track.lastOcrAttemptAt < retryDelay) continue;

        const minReadableWidth = Math.max(40, s.minCropWidth || INITIAL_SETTINGS.minCropWidth);
        const canReadFirstFrame = track.framesSeen >= 2 || track.bbox.confidence >= 0.60;
        if (!canReadFirstFrame || track.bbox.width < minReadableWidth) {
          track.ocrState = 'COLLECTING';
          continue;
        }

        const bestFrameEntry = globalBestFrameSelector.getBestCrop(track.trackNumber);
        const targetCropFromBest = !!bestFrameEntry?.canvas;
        const targetCrop = bestFrameEntry?.canvas ?? cropCanvasRegionFast(canvas, track.bbox);
        const qualityReport = bestFrameEntry?.quality || { overallScore: 0.6, recommendation: 'MARGINAL' as const };
        // Only skip truly unusable frames — MARGINAL frames go through to OCR.
        // The best-frame selector has already picked the sharpest available crop.
        if (qualityReport.recommendation === 'REJECT') {
          track.ocrState = 'LOW QUALITY';
          continue;
        }

        track.ocrRunning = true;
        track.ocrJobQueued = true;
        track.lastOcrAttemptAt = now;
        track.ocrState = 'OCR_RUNNING';
        activeOcrCount.current++;

        (async () => {
          const transientCanvases: HTMLCanvasElement[] = [];
          const rememberTransientCanvas = (crop?: HTMLCanvasElement | null) => {
            if (crop && !transientCanvases.includes(crop)) {
              transientCanvases.push(crop);
            }
          };

          try {
            if (!targetCropFromBest) rememberTransientCanvas(targetCrop);

            const innerTextCrop = createInnerPlateTextCrop(targetCrop);
            rememberTransientCanvas(innerTextCrop);

            const innerCrops = generateAdaptiveCrops(
              innerTextCrop,
              { x: 0, y: 0, width: innerTextCrop.width, height: innerTextCrop.height, confidence: 1.0 },
              360,
              108,
              ['ORIGINAL', 'INVERTED']
            );
            innerCrops.forEach(crop => {
              rememberTransientCanvas(crop.canvas);
              rememberTransientCanvas(crop.topLineCanvas);
              rememberTransientCanvas(crop.bottomLineCanvas);
            });

            const activeTrackLoad = Math.max(1, confirmedTracks.length);
            const fullCropVariants = activeTrackLoad >= 3
              ? ['ORIGINAL', 'DARK_BG', 'INVERTED'] as const
              : ['ORIGINAL', 'SHARPEN', 'DARK_BG', 'INVERTED'] as const;
            const adaptiveCrops = generateAdaptiveCrops(
              targetCrop,
              { x: 0, y: 0, width: targetCrop.width, height: targetCrop.height, confidence: 1.0 },
              360,
              108,
              [...fullCropVariants]
            );
            adaptiveCrops.forEach(crop => {
              rememberTransientCanvas(crop.canvas);
              rememberTransientCanvas(crop.topLineCanvas);
              rememberTransientCanvas(crop.bottomLineCanvas);
            });

            const maxCandidateCrops = activeTrackLoad >= 3 ? 4 : 6;
            const candidateCrops = [...innerCrops, ...adaptiveCrops].slice(0, maxCandidateCrops);
            const fallbackCrop = {
              canvas: targetCrop,
              isTwoLine: targetCrop.width / Math.max(1, targetCrop.height) < 2.3,
            };
            const expectedMinChars = getExpectedMinPlateChars(targetCrop);

            let text = '';
            let conf = 0;
            let bestScore = 0;
            let bestPatternValid = false;

            for (const crop of candidateCrops.length > 0 ? candidateCrops : [fallbackCrop]) {
              const result = await recognizePlateFromCanvas(crop.canvas, crop.isTwoLine);
              const resultText = result.normalizedPlate || result.text;
              const pattern = validateMalaysianPattern(resultText);
              const isPlausible = isPlausiblePlateCandidate(resultText, expectedMinChars, pattern.score);
              const lengthScore = Math.min(1.0, resultText.length / Math.max(expectedMinChars + 1, 6));
              const plausibilityPenalty = isPlausible ? 0 : 0.75;
              const score =
                result.confidence * 0.45 +
                pattern.score * 0.30 +
                lengthScore * 0.20 +
                (pattern.isValid ? 0.10 : 0) -
                plausibilityPenalty;

              if (resultText && isPlausible && score >= bestScore) {
                text = resultText;
                conf = result.confidence;
                bestScore = score;
                bestPatternValid = pattern.isValid;
              }

              if (resultText && isPlausible && result.confidence >= 0.25 && pattern.score >= 0.55 && score >= 0.58) {
                break;
              }
            }

            const updatedTrack = trackerRef.current.getTrack(trackId);
            if (!updatedTrack || updatedTrack.cooldownActive) return;

            // Lower gate: PP-OCR softmax confidence on small/blurry crops
            // can legitimately be 0.25–0.44 — these are valid reads that should
            // accumulate into consensus rather than being discarded.
            if (text && conf >= 0.25) {
              addOcrVoteToTrack(updatedTrack, text, conf, qualityReport.overallScore);
              updatedTrack.ocrState = 'CONSENSUS_BUILDING';

              const { digits } = countPlateChars(text);
              const canFastMatch = bestPatternValid && digits >= (text.length >= 5 ? 2 : 1);
              const veryStrongRead = canFastMatch && conf >= Math.max(0.60, s.recognitionThreshold) && bestScore >= 0.70;
              const strongRead = canFastMatch && conf >= 0.32 && bestScore >= 0.56;
              const requiredVotes = veryStrongRead ? 1 : strongRead ? Math.min(2, s.consensusVotes) : s.consensusVotes;
              const confidenceGate = veryStrongRead
                ? Math.min(s.recognitionThreshold, 0.58)
                : strongRead
                  ? Math.min(s.recognitionThreshold, 0.45)
                  : s.recognitionThreshold;
              const consensus = evaluateConsensus(updatedTrack, requiredVotes, confidenceGate);
              if (consensus.isStabilized) {
                const matchConfidence = Math.max(consensus.confidence, Math.min(0.98, bestScore));
                updatedTrack.stabilizedPlate = consensus.normalizedPlate;
                updatedTrack.stabilizedConfidence = matchConfidence;
                const finalOutcomeReady = canCommitFinalPlateOutcome(consensus.normalizedPlate);
                await runDatabaseMatch(updatedTrack, consensus.normalizedPlate, matchConfidence, {
                  commitNoCase: finalOutcomeReady,
                });
              }
            } else {
              if (updatedTrack.votes.size === 0) updatedTrack.ocrState = 'COLLECTING';
            }
          } catch (e) {
            console.warn('[OCR] Error:', e);
          } finally {
            transientCanvases.forEach(releaseCanvasMemory);
            const t = trackerRef.current.getTrack(trackId);
            if (t) {
              t.ocrRunning = false;
              t.ocrJobQueued = false;
              t.lastOcrCompletedAt = Date.now();
            }
            activeOcrCount.current = Math.max(0, activeOcrCount.current - 1);
          }
        })();
      }
    };

    const renderLoop = () => {
      if (!isPausedRef.current) {
        const video = videoRef.current;
        const overlayCanvas = canvasRef.current;

        if (video && overlayCanvas && video.readyState >= 2 && video.videoWidth > 0) {
          if (overlayCanvas.width !== video.videoWidth || overlayCanvas.height !== video.videoHeight) {
            overlayCanvas.width = video.videoWidth;
            overlayCanvas.height = video.videoHeight;
          }

          const overlayCtx = overlayCanvas.getContext('2d');
          if (overlayCtx) {
            overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
            const s = settingsRef.current;
            const displayTracks = s.debugMode
              ? trackerRef.current.getActiveTracks(false)
              : trackerRef.current.getActiveTracks(true);
            drawOverlays(overlayCtx, overlayCanvas.width, overlayCanvas.height, displayTracks, s.showCenterGuide, s.debugMode);
          }

          if (video.currentTime !== lastVideoTime) {
            camFrameCount.current++;
            lastVideoTime = video.currentTime;
          }
        }

        const now = Date.now();
        if (now - lastFpsTs.current >= 1000) {
          setCamFps(camFrameCount.current);
          camFrameCount.current = 0;
          lastFpsTs.current = now;
        }
      }
      animId = requestAnimationFrame(renderLoop);
    };

    animId = requestAnimationFrame(renderLoop);
    scheduleDetection(100);

    return () => {
      stopped = true;
      cancelAnimationFrame(animId);
      if (detectionTimeout) clearTimeout(detectionTimeout);
    };
  }, [cameraReady, runDatabaseMatch]);

  // ─── 8. Canvas Overlay Drawing ────────────────────────────────────────────
  function drawOverlays(
    ctx: CanvasRenderingContext2D,
    W: number,
    H: number,
    tracks: ActiveTrack[],
    showGuide: boolean,
    debug: boolean
  ) {
    if (showGuide) {
      ctx.strokeStyle = 'rgba(0, 216, 246, 0.2)';
      ctx.lineWidth = 1;
      ctx.setLineDash([6, 4]);
      const gW = W * 0.6, gH = gW / 4;
      ctx.strokeRect((W - gW) / 2, (H - gH) / 2, gW, gH);
      ctx.setLineDash([]);
    }

    tracks.forEach(track => {
      // Use smoothBbox for display so camera shake doesn't make boxes jitter.
      const { x, y, width, height } = track.smoothBbox;
      const color = getTrackColor(track);
      const label = getTrackStatusLabel(track);
      const reading = getTrackPlateText(track);
      const angle = track.overlayAngle ?? 0;
      const cx = x + width / 2;
      const cy = y + height / 2;
      const boxW = width * 1.02;
      const boxH = height * 1.08;

      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(angle);
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.setLineDash([]);
      ctx.shadowColor = color;
      ctx.shadowBlur = 8;
      ctx.beginPath();
      drawRoundedRect(ctx, -boxW / 2, -boxH / 2, boxW, boxH, 7);
      ctx.stroke();
      ctx.restore();

      const labelText = reading || label;

      if (labelText) {
        ctx.save();
        ctx.font = 'bold 14px sans-serif';
        const textW = ctx.measureText(labelText).width;
        const pillW = Math.max(56, textW + 18);
        const pillH = 26;
        const normalX = Math.sin(angle);
        const normalY = -Math.cos(angle);
        const pillCx = clampNumber(cx + normalX * (boxH / 2 + pillH / 2 + 6), pillW / 2 + 2, W - pillW / 2 - 2);
        const pillCy = clampNumber(cy + normalY * (boxH / 2 + pillH / 2 + 6), pillH / 2 + 2, H - pillH / 2 - 2);
        const pillX = pillCx - pillW / 2;
        const pillY = pillCy - pillH / 2;

        ctx.shadowColor = color;
        ctx.shadowBlur = 10;
        ctx.fillStyle = color;
        ctx.beginPath();
        drawRoundedRect(ctx, pillX, pillY, pillW, pillH, 6);
        ctx.fill();

        ctx.shadowBlur = 0;
        ctx.fillStyle = track.matchType === 'EXACT' ? '#ffffff' : '#061018';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(labelText, pillCx, pillCy);
        ctx.restore();
      }
    });
  }

  // ─── 9. Match Actions ─────────────────────────────────────────────────────
  const confirmVehicle = async (entry: MatchEntry) => {
    if (entry.scanId) {
      await fetch('/api/scans', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: entry.scanId, action: 'CONFIRM' }),
      });
    }
    setMatchQueue(q => q.filter(m => m.plate !== entry.plate));
    setViewingMatch(null);
  };

  const reportWrong = async (entry: MatchEntry) => {
    if (entry.scanId) {
      await fetch('/api/scans', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: entry.scanId, action: 'REPORT_WRONG' }),
      });
    }
    setMatchQueue(q => q.filter(m => m.plate !== entry.plate));
    setViewingMatch(null);
  };

  const activeMatches = matchQueue.filter(m => !m.dismissed);
  const isReadingPlate = cameraReady && !cameraError && tracksList.some(
    t => ['COLLECTING', 'OCR_RUNNING', 'OCR RUNNING', 'CONSENSUS_BUILDING', 'CONSENSUS', 'DB_CHECKING', 'DATABASE CHECK'].includes(t.ocrState)
  );
  const runtimeReady = isRuntimeScanningReady(runtimeState);
  const bottomStatus = cameraError
    ? { label: 'CAMERA UNAVAILABLE', dotClass: 'bg-rose-500', textClass: 'text-rose-400' }
    : !cameraReady
      ? { label: 'STARTING CAMERA', dotClass: 'bg-slate-500 animate-pulse', textClass: 'text-slate-400' }
      : !runtimeReady
        ? { label: 'AI WARMING UP', dotClass: 'bg-[#00d8f6] animate-pulse', textClass: 'text-[#00d8f6]' }
      : isPaused
        ? { label: 'SCANNER PAUSED', dotClass: 'bg-slate-500', textClass: 'text-slate-400' }
        : isReadingPlate
          ? { label: 'READING PLATE', dotClass: 'bg-[#00d8f6] animate-pulse', textClass: 'text-[#00d8f6]' }
          : { label: 'SCANNING SCENE', dotClass: 'bg-amber-500 animate-pulse', textClass: 'text-amber-500' };
  const resultTrack = tracksList.find(t => t.stabilizedPlate)
    || tracksList.find(t => getTrackPlateText(t))
    || null;
  const resultPlateText = resultTrack ? getTrackPlateText(resultTrack) : '';
  const hasResultCard = cameraReady && !cameraError && !!resultTrack && !!resultPlateText;
  const resultStatus = resultTrack?.matchType === 'EXACT'
    ? { label: 'Match Found', chipClass: 'bg-rose-500/15 text-rose-300 border-rose-400/40', iconClass: 'text-rose-300 bg-rose-500/15' }
    : resultTrack?.matchType === 'POSSIBLE'
      ? { label: 'Possible', chipClass: 'bg-amber-500/15 text-amber-300 border-amber-400/40', iconClass: 'text-amber-300 bg-amber-500/15' }
      : resultTrack?.matchType === 'NONE'
        ? { label: 'No Case', chipClass: 'bg-emerald-500/15 text-emerald-300 border-emerald-400/40', iconClass: 'text-emerald-300 bg-emerald-500/15' }
        : { label: 'Active', chipClass: 'bg-[#00d8f6]/15 text-[#00d8f6] border-[#00d8f6]/40', iconClass: 'text-[#00d8f6] bg-[#00d8f6]/15' };

  return (
    <div className="fixed inset-0 bg-black flex flex-col overflow-hidden" style={{ zIndex: 100 }}>

      {/* ── TOP CONTROL BAR ── */}
      <div className="flex-shrink-0 flex items-center justify-between px-3 py-2 bg-black/80 backdrop-blur-md border-b border-white/10 z-20">
        <Link
          href="/"
          className="flex items-center gap-1.5 text-xs font-semibold text-slate-300 hover:text-white"
        >
          <ArrowLeft className="w-4 h-4" />
          <span className="hidden sm:inline">Back</span>
        </Link>

        <div className="flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-[#00d8f6] animate-ping" />
          <span className="text-xs sm:text-sm font-black tracking-widest text-white uppercase">
            Live ANPR Multi-Vehicle Scanner
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          {torchSupported && (
            <button
              onClick={toggleTorch}
              className={`p-2 rounded-lg border text-xs transition-colors ${torchOn ? 'bg-amber-500/20 text-amber-400 border-amber-500/30' : 'bg-white/5 text-slate-400 border-white/10 hover:text-white'}`}
              aria-label="Torch"
            >
              {torchOn ? <Zap className="w-4 h-4" /> : <ZapOff className="w-4 h-4" />}
            </button>
          )}
          {devices.length > 1 && (
            <button
              onClick={handleSwitchCamera}
              className="p-2 bg-white/5 border border-white/10 rounded-lg text-slate-300 hover:text-white"
              aria-label="Switch Camera"
            >
              <SwitchCamera className="w-4 h-4" />
            </button>
          )}
          <button
            onClick={togglePause}
            className={`p-2 rounded-lg border transition-colors ${isPaused ? 'bg-[#00d8f6]/20 text-[#00d8f6] border-[#00d8f6]/30' : 'bg-white/5 text-slate-300 border-white/10 hover:text-white'}`}
            aria-label={isPaused ? 'Resume' : 'Pause'}
          >
            {isPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
          </button>
          <Link
            href="/settings"
            className="p-2 bg-white/5 border border-white/10 rounded-lg text-slate-300 hover:text-white"
            aria-label="Settings"
          >
            <SettingsIcon className="w-4 h-4" />
          </Link>
        </div>
      </div>

      {/* ── MODEL STATUS BANNER (Only renders on error/degraded or if debugMode is true) ── */}
      {!cameraError && (
        <div className="px-3 py-1.5 z-20">
          <ModelStatusBanner
            runtimeState={runtimeState}
            detectorProvider={detectorProvider}
            ocrProvider={ocrProvider}
            benchmark={benchmarkResult}
            errorMessage={runtimeErrorMessage}
            debugMode={isDebugMode}
            onRetry={startRuntimeInit}
            onManualSearch={() => window.location.href = '/search'}
          />
        </div>
      )}

      {/* ── DEBUG PERFORMANCE CHIP (Strictly gated behind debugMode === true) ── */}
      {isDebugMode && (
        <div className="px-3 py-1 z-20 flex items-center gap-1.5 text-[10px] font-mono font-bold pointer-events-none">
          <span className="px-2 py-0.5 bg-black/70 text-[#00d8f6] rounded border border-[#00d8f6]/30">
            CAM {camFps} FPS
          </span>
          <span className="px-2 py-0.5 bg-black/70 text-amber-400 rounded border border-amber-400/30">
            DET {detFps} FPS
          </span>
          <span className="px-2 py-0.5 bg-black/70 text-emerald-400 rounded border border-emerald-400/30">
            PLATES {platesVisible}
          </span>
          <span className="px-2 py-0.5 bg-black/70 text-slate-300 rounded border border-slate-700">
            TRACKS {activeTracksCount}
          </span>
        </div>
      )}

      {/* ── MAIN CAMERA CANVAS AREA ── */}
      <div className="flex-1 relative overflow-hidden">
        {/* TOP MATCH / READING BANNER */}
        {(() => {
          const activeWithPlate = tracksList.find(t => t.isConfirmed && t.stabilizedPlate) || tracksList.find(t => t.stabilizedPlate);
          if (activeWithPlate && activeWithPlate.stabilizedPlate) {
            const badgeBg = activeWithPlate.matchType === 'EXACT'
              ? 'bg-rose-950/90 text-rose-300 border-rose-500/50'
              : activeWithPlate.matchType === 'POSSIBLE'
                ? 'bg-amber-950/90 text-amber-300 border-amber-500/50'
                : activeWithPlate.matchType === 'NONE'
                  ? 'bg-slate-900/90 text-slate-300 border-slate-700'
                  : 'bg-[#062936]/90 text-[#00d8f6] border-[#00d8f6]/40';

            const statusLabel = activeWithPlate.matchType === 'EXACT'
              ? 'REPO MATCH FOUND'
              : activeWithPlate.matchType === 'POSSIBLE'
                ? 'POSSIBLE MATCH'
                : activeWithPlate.matchType === 'NONE'
                  ? 'NO MATCH IN DATABASE'
                  : 'READING PLATE';
            const statusIcon = activeWithPlate.matchType === 'EXACT'
              ? '!'
              : activeWithPlate.matchType === 'POSSIBLE'
                ? '!'
                : activeWithPlate.matchType === 'NONE'
                  ? '✓'
                  : '•';

            return (
              <div className="absolute top-6 left-1/2 -translate-x-1/2 z-30 transition-all duration-300 pointer-events-none">
                <div className={`px-5 py-2.5 rounded-full font-semibold text-xs shadow-2xl border backdrop-blur-md flex items-center gap-2 ${badgeBg}`}>
                   <span className="font-bold">{statusIcon} {statusLabel}</span>
                   <span className="opacity-40">|</span>
                   <span className="font-mono font-extrabold text-white text-sm tracking-wider">{activeWithPlate.stabilizedPlate}</span>
                </div>
              </div>
            );
          }
          return null;
        })()}

        {/* BOTTOM SCANNING STATUS PILL */}
        {cameraReady && !cameraError && runtimeReady && (
          <div className={`absolute ${hasResultCard ? (trayExpanded ? 'bottom-48' : 'bottom-32') : 'bottom-6'} left-1/2 -translate-x-1/2 z-30 transition-all duration-300 pointer-events-none`}>
            {(() => {
              if (isReadingPlate) {
                return (
                  <div className="bg-[#00d8f6]/95 text-slate-950 px-6 py-2.5 rounded-full font-bold text-xs shadow-2xl border border-[#00d8f6] backdrop-blur-md flex items-center gap-2 animate-pulse">
                    <span className="w-2 h-2 rounded-full bg-slate-950 animate-ping" />
                    <span>Reading plate...</span>
                  </div>
                );
              }
              return (
                <div className="bg-[#1a1c23]/95 text-slate-200 px-6 py-2.5 rounded-full font-semibold text-xs shadow-2xl border border-white/10 backdrop-blur-md flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                  <span>Scanning scene for plates...</span>
                </div>
              );
            })()}
          </div>
        )}
        {cameraError && (
          <div className="absolute inset-0 flex items-center justify-center bg-[#090a0f] z-10 p-6">
            <div className="max-w-sm text-center">
              <AlertOctagon className="w-12 h-12 text-rose-500 mx-auto mb-3" />
              <h3 className="text-lg font-bold text-white mb-2">Camera Error</h3>
              <p className="text-xs text-slate-400 mb-4">{cameraError}</p>
              <button
                onClick={() => initCamera()}
                className="px-5 py-2.5 bg-[#00d8f6] text-slate-950 text-xs font-bold rounded-xl"
              >
                Retry Camera
              </button>
            </div>
          </div>
        )}

        {!cameraError && !cameraReady && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#090a0f] z-10 gap-4">
            <div className="w-12 h-12 border-4 border-[#00d8f6]/20 border-t-[#00d8f6] rounded-full animate-spin" />
            <p className="text-sm font-semibold text-slate-400">Loading YOLOv8 Malaysian Plate Detector…</p>
          </div>
        )}

        <video
          ref={videoRef}
          playsInline
          muted
          className="absolute inset-0 w-full h-full object-cover"
          style={{ display: cameraReady && !cameraError ? 'block' : 'none' }}
        />

        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full object-cover pointer-events-none"
          style={{ display: cameraReady ? 'block' : 'none' }}
        />

        {isPaused && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/60 z-10">
            <div className="flex flex-col items-center gap-2">
              <Pause className="w-14 h-14 text-white/60" />
              <span className="text-lg font-bold text-white/70">Paused</span>
              <button
                onClick={togglePause}
                className="mt-2 px-6 py-2.5 bg-[#00d8f6] text-slate-950 font-bold text-sm rounded-xl"
              >
                Resume Scanning
              </button>
            </div>
          </div>
        )}

        {/* MATCH NOTIFICATION BADGES */}
        {activeMatches.length > 0 && !viewingMatch && (
          <div className="absolute top-3 right-3 z-20 flex flex-col gap-2">
            {activeMatches.map(entry => (
              <button
                key={entry.plate}
                onClick={() => setViewingMatch(entry)}
                className={`flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-bold shadow-2xl border backdrop-blur-sm transition-all hover:scale-105 ${
                  entry.type === 'EXACT'
                    ? 'bg-rose-600/90 border-rose-500 text-white shadow-rose-900/60'
                    : 'bg-amber-500/90 border-amber-400 text-slate-950 shadow-amber-900/60'
                }`}
              >
                {entry.type === 'EXACT' ? (
                  <AlertOctagon className="w-4 h-4 shrink-0" />
                ) : (
                  <AlertTriangle className="w-4 h-4 shrink-0" />
                )}
                <span className="font-mono">{entry.plate}</span>
                <span>{entry.type === 'EXACT' ? 'MATCH' : 'POSSIBLE'}</span>
              </button>
            ))}
          </div>
        )}

        {hasResultCard && resultTrack && (
          <div className="absolute bottom-3 left-3 right-3 z-30 pointer-events-auto">
            <div className="rounded-2xl bg-[#101116]/95 border border-white/10 shadow-2xl backdrop-blur-md overflow-hidden">
              <button
                onClick={() => setTrayExpanded(v => !v)}
                className="w-full flex items-center gap-3 px-4 py-3 text-left"
              >
                <div className={`w-11 h-11 rounded-xl flex items-center justify-center shrink-0 ${resultStatus.iconClass}`}>
                  {resultTrack.matchType === 'EXACT' ? (
                    <AlertOctagon className="w-6 h-6" />
                  ) : (
                    <AlertTriangle className="w-6 h-6" />
                  )}
                </div>
                <div className="min-w-0 flex-1">
                  <div className="font-mono text-xl font-black tracking-widest text-white leading-tight">
                    {resultPlateText}
                  </div>
                  <div className={`mt-1 inline-flex items-center px-2.5 py-1 rounded-full border text-xs font-black ${resultStatus.chipClass}`}>
                    {resultStatus.label}
                  </div>
                </div>
                <div className="text-slate-500 shrink-0">
                  {trayExpanded ? <ChevronDown className="w-5 h-5" /> : <ChevronUp className="w-5 h-5" />}
                </div>
              </button>

              {trayExpanded && (resultTrack.matchedVehicle || (resultTrack.possibleMatchVehicles?.length ?? 0) > 0) && (
                <div className="border-t border-white/10 px-4 pb-3 text-xs">
                  {resultTrack.matchedVehicle && (
                    <div className="grid grid-cols-2 gap-2 pt-3">
                      <div className="rounded-lg bg-black/30 p-2">
                        <div className="text-slate-500">Customer</div>
                        <div className="text-white font-bold truncate">{resultTrack.matchedVehicle.customerName}</div>
                      </div>
                      <div className="rounded-lg bg-black/30 p-2">
                        <div className="text-slate-500">Vehicle</div>
                        <div className="text-white font-bold truncate">
                          {resultTrack.matchedVehicle.vehicleMake} {resultTrack.matchedVehicle.vehicleModel}
                        </div>
                      </div>
                    </div>
                  )}

                  {!resultTrack.matchedVehicle && (resultTrack.possibleMatchVehicles?.length ?? 0) > 0 && (
                    <div className="pt-3 space-y-2">
                      {resultTrack.possibleMatchVehicles?.slice(0, 2).map(v => (
                        <div key={v.id} className="rounded-lg bg-black/30 p-2 flex items-center justify-between gap-2">
                          <span className="font-mono font-bold text-white">{v.plateNumber}</span>
                          <span className="text-slate-300 truncate">{v.vehicleMake} {v.vehicleModel}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* ── NEW BOTTOM NAVIGATION BAR (Mockup Layout) ── */}
      <div className="flex-shrink-0 bg-[#090a0f] border-t border-[#252833] flex flex-col">
        {/* STATUS INDICATOR */}
        <div className="flex items-center gap-3 px-6 py-4 border-b border-[#252833]">
          <div className={`w-2.5 h-2.5 rounded-full ${bottomStatus.dotClass}`} />
          <span className={`${bottomStatus.textClass} text-xs font-bold tracking-widest uppercase`}>
            {bottomStatus.label}
          </span>
        </div>
        
        {/* TAB BAR */}
        <div className="flex items-center justify-around py-3">
          <Link href="/" className="flex flex-col items-center gap-1.5 p-2 text-slate-500 hover:text-white transition-colors">
            <LayoutGrid className="w-6 h-6" />
            <span className="text-[10px] font-medium">Dashboard</span>
          </Link>
          <Link href="/search" className="flex flex-col items-center gap-1.5 p-2 text-slate-500 hover:text-white transition-colors">
            <Search className="w-6 h-6" />
            <span className="text-[10px] font-medium">Search</span>
          </Link>
          <div className="flex flex-col items-center gap-1.5 p-2 text-[#00d8f6]">
            <Camera className="w-6 h-6" />
            <span className="text-[10px] font-bold">Scanner</span>
          </div>
          <Link href="/manage" className="flex flex-col items-center gap-1.5 p-2 text-slate-500 hover:text-white transition-colors">
            <Car className="w-6 h-6" />
            <span className="text-[10px] font-medium">Manage</span>
          </Link>
        </div>
      </div>

      {/* ── MATCH DETAIL MODAL ── */}
      {viewingMatch && (
        <div className="absolute inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-4">
          <div
            className={`max-w-md w-full rounded-2xl p-6 shadow-2xl border-2 ${
              viewingMatch.type === 'EXACT'
                ? 'bg-rose-950/95 border-rose-600'
                : 'bg-amber-950/95 border-amber-500'
            }`}
          >
            <div className="flex items-center justify-between mb-4 pb-4 border-b border-white/10">
              <div className="flex items-center gap-3">
                {viewingMatch.type === 'EXACT'
                  ? <AlertOctagon className="w-7 h-7 text-rose-400 animate-bounce" />
                  : <AlertTriangle className="w-7 h-7 text-amber-400" />}
                <div>
                  <div className="text-[10px] font-black text-white/60 tracking-widest uppercase">Track #{viewingMatch.trackId.replace('trk-', '')}</div>
                  <h2 className="text-xl font-black text-white">
                    {viewingMatch.type === 'EXACT' ? 'MATCH FOUND' : 'POSSIBLE MATCH'}
                  </h2>
                </div>
              </div>
              <span className="text-xl font-mono font-black text-white bg-black/50 px-3 py-1 rounded-xl border border-white/20">
                {viewingMatch.plate}
              </span>
            </div>

            {viewingMatch.type === 'EXACT' && viewingMatch.vehicle && (
              <div className="grid grid-cols-2 gap-2 mb-4 text-xs">
                {[
                  ['Pelanggan', viewingMatch.vehicle.customerName],
                  ['Kenderaan', `${viewingMatch.vehicle.vehicleMake} ${viewingMatch.vehicle.vehicleModel} (${viewingMatch.vehicle.vehicleColor})`],
                  ['Syarikat Kewangan', viewingMatch.vehicle.financeCompany],
                  ['Rujukan Kes', viewingMatch.vehicle.caseReference],
                ].map(([label, value]) => (
                  <div key={label} className="bg-black/50 p-2.5 rounded-lg border border-white/5">
                    <span className="text-slate-400 block mb-0.5">{label}</span>
                    <span className="text-white font-semibold">{value}</span>
                  </div>
                ))}
                <div className="col-span-2 bg-black/50 p-2.5 rounded-lg border border-rose-900/40">
                  <span className="text-slate-400 block mb-0.5">Jumlah Tunggakan</span>
                  <span className="text-rose-400 font-black text-lg">
                    RM {viewingMatch.vehicle.outstandingAmount.toLocaleString('ms-MY', { minimumFractionDigits: 2 })}
                  </span>
                </div>
              </div>
            )}

            {viewingMatch.type === 'POSSIBLE' && viewingMatch.possibleMatches.length > 0 && (
              <div className="mb-4 text-xs space-y-2">
                <p className="text-amber-200 mb-2">
                  Nombor plat <span className="font-mono font-bold text-white">{viewingMatch.plate}</span> hampir sepadan. Sahkan secara visual.
                </p>
                {viewingMatch.possibleMatches.map(v => (
                  <div key={v.id} className="bg-black/50 p-2.5 rounded-lg border border-amber-900/30">
                    <div className="flex justify-between items-center">
                      <span className="font-mono font-bold text-white">{v.plateNumber}</span>
                      <span className="text-rose-400 font-bold">RM {v.outstandingAmount.toLocaleString('ms-MY', { minimumFractionDigits: 2 })}</span>
                    </div>
                    <div className="text-slate-300 mt-0.5">{v.customerName} · {v.vehicleMake} {v.vehicleModel} ({v.vehicleColor})</div>
                  </div>
                ))}
              </div>
            )}

            <div className="text-[10px] text-slate-400 mb-4">
              Confidence: {Math.round(viewingMatch.confidence * 100)}% · Multi-vehicle scanner is running in background.
            </div>

            <div className="flex flex-wrap gap-2 justify-end">
              <button
                onClick={() => reportWrong(viewingMatch)}
                className="px-3 py-2 bg-black/60 text-slate-300 border border-white/10 rounded-xl text-xs font-semibold hover:bg-black"
              >
                Report Wrong Reading
              </button>
              <button
                onClick={() => { setViewingMatch(null); }}
                className="px-4 py-2 bg-slate-700 text-white rounded-xl text-xs font-semibold hover:bg-slate-600"
              >
                Continue Scanning
              </button>
              {viewingMatch.type === 'EXACT' && (
                <button
                  onClick={() => confirmVehicle(viewingMatch)}
                  className="px-5 py-2 bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-black rounded-xl text-xs"
                >
                  Confirm Vehicle
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
