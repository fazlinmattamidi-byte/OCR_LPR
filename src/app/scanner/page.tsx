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
  initLocalOnnxSession,
  validateDetector,
  getActiveDetectorProvider,
  DetectedPlateBox,
  ActiveExecutionProvider,
} from '@/lib/anpr/yoloDetector';
import {
  generateAdaptiveCrops,
  cropCanvasRegion,
  prioritiseTracks,
} from '@/lib/anpr/imageProcessor';
import { globalBestFrameSelector } from '@/lib/anpr/bestFrameSelector';
import { recognizePlateFromCanvas } from '@/lib/anpr/ocrEngine';
import { addOcrVoteToTrack, evaluateConsensus, getTrackReadingDisplay } from '@/lib/anpr/consensus';
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
import { initPpOcrSession, getActivePpOcrProvider, ActiveOcrProvider } from '@/lib/anpr/ppOcrEngine';

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
  if (track.matchType === 'NONE') return '#10b981';    // green
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
  const trackerRef = useRef<PlateTracker>(new PlateTracker(20, 8));
  const streamRef = useRef<MediaStream | null>(null);

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

  const resetLiveScanUi = useCallback(() => {
    trackerRef.current.clear();
    globalBestFrameSelector.resetAll();
    camFrameCount.current = 0;
    detFrameCount.current = 0;
    activeOcrCount.current = 0;
    setTracksList([]);
    setPlatesVisible(0);
    setActiveTracksCount(0);
    setCamFps(0);
    setDetFps(0);
  }, []);

  const startRuntimeInit = useCallback(async () => {
    setRuntimeState('LOADING_MODELS');
    const res = await initializeANPRRuntime();
    setRuntimeState(res.state);
    if (res.benchmark) setBenchmarkResult(res.benchmark);
    setRuntimeErrorMessage(getRuntimeErrorMessage());
    setDetectorProvider(getActiveDetectorProvider());
    setOcrProvider(getActivePpOcrProvider());
  }, []);

  // ─── 1. Load Settings & Initialize Runtime ───────────────────────────
  useEffect(() => {
    fetch('/api/settings')
      .then(r => r.json())
      .then(data => {
        if (data.success && data.settings) {
          settingsRef.current = { ...settingsRef.current, ...data.settings };
          setIsDebugMode(!!data.settings.debugMode);
          trackerRef.current.setLostTrackTimeout(data.settings.lostTrackTimeout ?? 20);
        }
      })
      .catch(() => {});

    startRuntimeInit();
  }, [startRuntimeInit]);

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
  const runDatabaseMatch = useCallback(async (track: ActiveTrack, plate: string, confidence: number) => {
    const cooldownMs = settingsRef.current.duplicateCooldown * 1000;
    const now = Date.now();
    const lastSearch = cooldownMap.current.get(plate) ?? 0;

    if (now - lastSearch < cooldownMs) return;
    cooldownMap.current.set(plate, now);

    track.ocrState = 'DB_CHECKING';

    try {
      const searchRes = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ plateNumber: plate, source: 'CAMERA', confidence }),
      }).then(r => r.json());

      if (!searchRes.success) return;

      const scanRes = await fetch('/api/scans', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          detectedPlate: plate,
          normalizedPlate: plate,
          confidence,
          matchType: searchRes.matchType,
          matchedVehicleId: searchRes.matchedVehicle?.id ?? undefined,
          source: 'CAMERA',
          trackId: track.trackId,
          frameCount: track.framesSeen,
          firstSeenAt: new Date(Date.now() - track.framesSeen * 33).toISOString(),
        }),
      }).then(r => r.json());

      track.matchType = searchRes.matchType === 'EXACT' ? 'EXACT' : searchRes.matchType === 'POSSIBLE' ? 'POSSIBLE' : 'NONE';
      track.matchedVehicle = searchRes.matchedVehicle ?? undefined;
      track.possibleMatchVehicles = searchRes.possibleMatches ?? [];
      track.ocrState = track.matchType === 'EXACT' ? 'MATCHED' : track.matchType === 'POSSIBLE' ? 'POSSIBLE MATCH' : 'NO CASE';

      if (searchRes.matchType === 'EXACT') {
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
        setMatchQueue(q => [...q.filter(m => m.plate !== plate), entry]);

      } else if (searchRes.matchType === 'POSSIBLE') {
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
        setMatchQueue(q => [...q.filter(m => m.plate !== plate), entry]);
      }

      track.ocrState = 'COOLDOWN';
      track.cooldownActive = true;
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
    let detectionTimeout: NodeJS.Timeout;
    let detTs = Date.now();
    let detCount = 0;

    const runDetection = async () => {
      if (!videoRef.current || !canvasRef.current || isPausedRef.current) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (video.readyState < 2 || video.videoWidth === 0) return;

      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) return;

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const s = settingsRef.current;

      // ── Step 1: Malaysian Plate Detector ──
      const detectedPlates = await detectMalaysianPlates(canvas, {
        minConfidence: s.detectionThreshold,
        enginePreference: s.detectorEngine,
        developerMode: s.debugMode,
      });

      if (detectedPlates.length > 0) {
        setActiveEngine(detectedPlates[0].sourceEngine);
        const avgConf = detectedPlates.reduce((sum, p) => sum + p.confidence, 0) / detectedPlates.length;
        setAvgConfidence(avgConf);
      } else {
        const val = await validateDetector();
        if (val.valid) setActiveEngine('LOCAL_ONNX');
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

      setPlatesVisible(bboxList.length);
      setActiveTracksCount(confirmedTracks.length); // Update metric to show confirmed count
      setTracksList([...displayTracks]); // UI List

      // ── Step 3: Best Frame Selection (Only on Confirmed Tracks) ──
      confirmedTracks.forEach(track => {
        if (track.ocrState === 'COOLDOWN' || track.ocrState === 'MATCHED') return;
        const cropCanvas = cropCanvasRegion(canvas, track.bbox);
        globalBestFrameSelector.addCropCandidate(track.trackNumber, cropCanvas, track.bbox);
      });

      // ── Step 4: Draw Overlays ──
      drawOverlays(ctx, canvas.width, canvas.height, displayTracks, s.showCenterGuide, s.debugMode);

      // ── Step 5: OCR Priority Queue (Async Decoupled) ──
      processOcrQueue(confirmedTracks, canvas, s);

      detCount++;
      const now = Date.now();
      if (now - detTs >= 1000) {
        setDetFps(detCount);
        detCount = 0;
        detTs = now;
      }

      // Adaptive FPS: If OCR is busy, reduce detector FPS to save memory/battery
      const nextDelay = activeOcrCount.current > 0 ? 166 : 100; // ~6 FPS if busy, 10 FPS if idle
      detectionTimeout = setTimeout(runDetection, nextDelay);
    };

    const processOcrQueue = async (confirmedTracks: ActiveTrack[], canvas: HTMLCanvasElement, s: ScannerSettings) => {
      const priorityIds = prioritiseTracks(
        confirmedTracks.map(t => ({ trackId: t.trackId, bbox: t.bbox, framesSeen: t.framesSeen, ocrState: t.ocrState })),
        canvas.width,
        canvas.height,
        s.maxOcrConcurrency
      );

      for (const trackId of priorityIds) {
        const track = trackerRef.current.getTrack(trackId);
        if (!track || !track.isConfirmed || track.ocrRunning || track.cooldownActive) continue;
        if (activeOcrCount.current >= s.maxOcrConcurrency) break;

        // OCR Gating: Need at least 2 frames and plate wide enough to read
        // 60px is the practical minimum for PP-OCR to extract characters on mobile.
        if (track.framesSeen < 2 || track.bbox.width < 60) {
          track.ocrState = 'COLLECTING';
          continue;
        }

        const bestFrameEntry = globalBestFrameSelector.getBestCrop(track.trackNumber);
        const targetCrop = bestFrameEntry ? bestFrameEntry.canvas : null;
        
        if (!targetCrop) continue; // Skip if no crop saved yet
        
        const qualityReport = bestFrameEntry?.quality || { overallScore: 0.6, recommendation: 'PASS' };
        // Only skip truly unusable frames — MARGINAL frames go through to OCR.
        // The best-frame selector has already picked the sharpest available crop.
        if (qualityReport.recommendation === 'REJECT') {
          track.ocrState = 'LOW QUALITY';
          continue;
        }

        track.ocrRunning = true;
        track.ocrState = 'OCR_RUNNING';
        activeOcrCount.current++;

        (async () => {
          try {
            // Memory optimization: generate Adaptive Crops only when needed
            const adaptiveCrops = generateAdaptiveCrops(targetCrop, {x:0, y:0, width: targetCrop.width, height: targetCrop.height, confidence: 1.0});
            const preferredVariants = ['ORIGINAL', 'SHARPEN', 'DARK_BG'] as const;
            const preferredCrops = preferredVariants
              .map(variant => adaptiveCrops.find(c => c.variant === variant))
              .filter((crop): crop is (typeof adaptiveCrops)[number] => Boolean(crop));
            const candidateCrops = [...preferredCrops, ...adaptiveCrops]
              .filter((crop, index, list) => crop && list.findIndex(item => item?.canvas === crop.canvas) === index)
              .slice(0, 4);
            const fallbackCrop = {
              canvas: targetCrop,
              isTwoLine: targetCrop.width / Math.max(1, targetCrop.height) < 2.3,
            };

            let text = '';
            let conf = 0;
            let bestScore = 0;

            for (const crop of candidateCrops.length > 0 ? candidateCrops : [fallbackCrop]) {
              const result = await recognizePlateFromCanvas(crop.canvas, crop.isTwoLine);
              const score = result.confidence * 0.65 + result.patternScore * 0.35;

              if (result.text && score >= bestScore) {
                text = result.text;
                conf = result.confidence;
                bestScore = score;
              }

              if (result.text && result.confidence >= 0.30 && result.patternScore >= 0.55) {
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

              const consensus = evaluateConsensus(updatedTrack, s.consensusVotes, s.recognitionThreshold);
              if (consensus.isStabilized) {
                updatedTrack.stabilizedPlate = consensus.normalizedPlate;
                updatedTrack.stabilizedConfidence = consensus.confidence;
                await runDatabaseMatch(updatedTrack, consensus.normalizedPlate, consensus.confidence);
              }
            } else {
              if (updatedTrack.votes.size === 0) updatedTrack.ocrState = 'COLLECTING';
            }
          } catch (e) {
            console.warn('[OCR] Error:', e);
          } finally {
            const t = trackerRef.current.getTrack(trackId);
            if (t) t.ocrRunning = false;
            activeOcrCount.current = Math.max(0, activeOcrCount.current - 1);
          }
        })();
      }
    };

    const renderLoop = () => {
      if (!isPausedRef.current) {
        camFrameCount.current++;
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
    detectionTimeout = setTimeout(runDetection, 100);

    return () => {
      cancelAnimationFrame(animId);
      clearTimeout(detectionTimeout);
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
      // smoothBbox is EMA-interpolated toward the raw detection each frame.
      const { x, y, width, height } = track.smoothBbox;
      const color = getTrackColor(track);
      const label = getTrackStatusLabel(track);
      const reading = track.stabilizedPlate || getTrackReadingDisplay(track);
      const displayNum = track.trackNumber;

      // NEW CLEAN UI DESIGN: White rounded box around plate
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
      ctx.lineWidth = 2.5;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.roundRect(x, y, width, height, 6);
      ctx.stroke();

      const labelText = reading || label;

      if (labelText) {
        ctx.font = 'bold 12px sans-serif';
        const textW = ctx.measureText(labelText).width;
        const pillW = textW + 20;
        const pillH = 26;
        const pillX = x + (width / 2) - (pillW / 2); // Centered above box
        const pillY = y - pillH - 8;

        // Dark semi-transparent pill background
        ctx.fillStyle = 'rgba(15, 15, 20, 0.85)';
        ctx.beginPath();
        ctx.roundRect(pillX, Math.max(2, pillY), pillW, pillH, 13);
        ctx.fill();

        // White text
        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(labelText, pillX + pillW / 2, Math.max(2, pillY) + pillH / 2);
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
  const bottomStatus = cameraError
    ? { label: 'CAMERA UNAVAILABLE', dotClass: 'bg-rose-500', textClass: 'text-rose-400' }
    : !cameraReady
      ? { label: 'STARTING CAMERA', dotClass: 'bg-slate-500 animate-pulse', textClass: 'text-slate-400' }
      : isPaused
        ? { label: 'SCANNER PAUSED', dotClass: 'bg-slate-500', textClass: 'text-slate-400' }
        : isReadingPlate
          ? { label: 'READING PLATE', dotClass: 'bg-[#00d8f6] animate-pulse', textClass: 'text-[#00d8f6]' }
          : { label: 'SCANNING SCENE', dotClass: 'bg-amber-500 animate-pulse', textClass: 'text-amber-500' };

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
            const isMatch = activeWithPlate.matchType === 'EXACT' || activeWithPlate.matchType === 'POSSIBLE';
            const badgeBg = isMatch
              ? (activeWithPlate.matchType === 'EXACT' ? 'bg-rose-950/90 text-rose-300 border-rose-500/50' : 'bg-amber-950/90 text-amber-300 border-amber-500/50')
              : 'bg-slate-900/90 text-slate-300 border-slate-700';

            const statusLabel = isMatch
              ? (activeWithPlate.matchType === 'EXACT' ? '🚨 REPO MATCH FOUND' : '⚠️ POSSIBLE MATCH')
              : '✓ NO MATCH IN DATABASE';

            return (
              <div className="absolute top-6 left-1/2 -translate-x-1/2 z-30 transition-all duration-300 pointer-events-none">
                <div className={`px-5 py-2.5 rounded-full font-semibold text-xs shadow-2xl border backdrop-blur-md flex items-center gap-2 ${badgeBg}`}>
                   <span className="font-bold">{statusLabel}</span>
                   <span className="opacity-40">|</span>
                   <span className="font-mono font-extrabold text-white text-sm tracking-wider">{activeWithPlate.stabilizedPlate}</span>
                </div>
              </div>
            );
          }
          return null;
        })()}

        {/* BOTTOM SCANNING STATUS PILL */}
        {cameraReady && !cameraError && (
          <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-30 transition-all duration-300 pointer-events-none">
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
          style={{ opacity: 0 }}
        />

        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full object-cover"
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
