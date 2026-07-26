'use client';

import React, { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { useLanguage } from '@/context/LanguageContext';
import { useStorage } from '@/context/StorageContext';
import { useAuth } from '@/context/AuthContext';
import { cleanPlateNumber, formatMYR } from '@/lib/utils';
import { DetectionLog, Vehicle } from '@/types';
import {
  Activity,
  ArrowLeft,
  BookmarkCheck,
  CheckCircle2,
  MapPin,
  Pause,
  Play,
  Plus,
  ShieldAlert,
  Video,
  Volume2,
  VolumeX,
  X,
  XCircle,
} from 'lucide-react';

const DEMO_PLATES = [
  'ANN7569',
  'W8821X',
  'VAA9911',
  'QAA1234',
  'ABC1234',
  'BKP4412',
  'PKK7788',
  'SAB1234',
  'AKR5050',
  'JQX2244',
];

type AlertMatch = {
  vehicle: Vehicle;
  cameraName: string;
  cameraId: string;
};

type CameraSlot = {
  id: string;
  deviceId: string;
};

type ScanLocation = {
  name: string;
  gps: string;
};

type SessionDetection = DetectionLog & ScanLocation;

type AudioWindow = Window &
  typeof globalThis & {
    webkitAudioContext?: typeof AudioContext;
  };

const SCAN_LOCATIONS: ScanLocation[] = [
  { name: 'Sungai Besi Toll Plaza', gps: '3.0602, 101.7047' },
  { name: 'Jalan Tun Razak, Kuala Lumpur', gps: '3.1618, 101.7165' },
  { name: 'Federal Highway KM12', gps: '3.0837, 101.6129' },
  { name: 'Shah Alam Section 13', gps: '3.0831, 101.5443' },
  { name: 'Penang Bridge Checkpoint', gps: '5.3674, 100.3422' },
];

const RECENT_DETECTIONS_STORAGE_KEY = 'track_recent_live_detections';

export default function ScannerPage() {
  const { t, language } = useLanguage();
  const { vehicles, addHistoryLog, updateVehicle } = useStorage();
  const { role } = useAuth();

  const videoRefs = useRef<Record<string, HTMLVideoElement | null>>({});
  const activeStreamsRef = useRef<Record<string, MediaStream>>({});
  const vehiclesRef = useRef(vehicles);
  const addHistoryLogRef = useRef(addHistoryLog);
  const soundEnabledRef = useRef(true);

  const [availableCameras, setAvailableCameras] = useState<MediaDeviceInfo[]>([]);
  const [cameraSlots, setCameraSlots] = useState<CameraSlot[]>([{ id: 'camera-slot-1', deviceId: '' }]);
  const [activeCameraSlotId, setActiveCameraSlotId] = useState('camera-slot-1');
  const [previewSlotIds, setPreviewSlotIds] = useState<string[]>([]);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [cameraError, setCameraError] = useState('');
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [currentPlate, setCurrentPlate] = useState('READY');
  const [lastDetectedSlotId, setLastDetectedSlotId] = useState('');
  const [activeAlertMatch, setActiveAlertMatch] = useState<AlertMatch | null>(null);
  const [liveDetections, setLiveDetections] = useState<SessionDetection[]>([]);
  const [expandedDetectionId, setExpandedDetectionId] = useState<string | null>(null);

  useEffect(() => {
    vehiclesRef.current = vehicles;
  }, [vehicles]);

  useEffect(() => {
    addHistoryLogRef.current = addHistoryLog;
  }, [addHistoryLog]);

  useEffect(() => {
    soundEnabledRef.current = soundEnabled;
  }, [soundEnabled]);

  useEffect(() => {
    try {
      const storedDetections = localStorage.getItem(RECENT_DETECTIONS_STORAGE_KEY);
      if (storedDetections) {
        const parsedDetections = JSON.parse(storedDetections) as SessionDetection[];
        setLiveDetections(parsedDetections.slice(0, 8));
        if (parsedDetections[0]) {
          setCurrentPlate(parsedDetections[0].plate);
          setLastDetectedSlotId(parsedDetections[0].cameraId);
        }
      }
    } catch {
      localStorage.removeItem(RECENT_DETECTIONS_STORAGE_KEY);
    }

    void refreshCameraList();

    return () => {
      stopCamera();
    };
  }, []);

  useEffect(() => {
    if (!isScanning) return;

    const timer = window.setInterval(() => {
      const randomPlate = DEMO_PLATES[Math.floor(Math.random() * DEMO_PLATES.length)];
      const scannedPlate = cleanPlateNumber(randomPlate);
      const matchedVehicle =
        vehiclesRef.current.find(
          (vehicle) => cleanPlateNumber(vehicle.plate) === scannedPlate && vehicle.status === 'ACTIVE'
        ) || null;
      const isMatch = Boolean(matchedVehicle);
      const confidence = Number((91 + Math.random() * 8).toFixed(1));
      const timestamp = new Date();
      const slotIndex = Math.floor(Math.random() * cameraSlots.length);
      const scanSlot = cameraSlots[slotIndex] || cameraSlots[0];
      const scanCameraName = getCameraSlotLabel(scanSlot, slotIndex);
      const scanLocation = SCAN_LOCATIONS[Math.floor(Math.random() * SCAN_LOCATIONS.length)];

      const newDetection: SessionDetection = {
        id: `det-${timestamp.getTime()}-${Math.floor(Math.random() * 1000)}`,
        plate: scannedPlate,
        confidence,
        matched: isMatch,
        vehicleId: matchedVehicle?.id,
        timestamp: timestamp.toLocaleTimeString('en-GB'),
        cameraId: scanSlot?.id || 'laptop-camera',
        cameraName: scanCameraName,
        name: scanLocation.name,
        gps: scanLocation.gps,
      };

      setCurrentPlate(scannedPlate);
      setLastDetectedSlotId(scanSlot?.id || '');
      setLiveDetections((prevStream) => {
        const nextStream = [newDetection, ...prevStream.slice(0, 7)];
        localStorage.setItem(RECENT_DETECTIONS_STORAGE_KEY, JSON.stringify(nextStream));
        return nextStream;
      });

      addHistoryLogRef.current({
        type: 'DETECTION',
        action: isMatch ? `Tanda Tindakan (Pengimbas): ${scannedPlate}` : `Live Scan: ${scannedPlate}`,
        plate: scannedPlate,
        details:
          isMatch && matchedVehicle
            ? `AI Confidence: ${confidence}% - Tanda Tindakan: ${matchedVehicle.brand} ${matchedVehicle.model} - Location: ${scanLocation.name} (${scanLocation.gps})`
            : `AI Confidence: ${confidence}% - No Match Found - Location: ${scanLocation.name} (${scanLocation.gps})`,
        note:
          isMatch && matchedVehicle
            ? `Match Found: ${matchedVehicle.brand} ${matchedVehicle.model} via ${scanCameraName} at ${scanLocation.name}`
            : `No match from ${scanCameraName} at ${scanLocation.name}`,
        userRole: role,
        statusMatch: isMatch ? 'EXACT' : 'NONE',
      });

      if (isMatch && matchedVehicle) {
        setActiveAlertMatch({ vehicle: matchedVehicle, cameraName: scanCameraName, cameraId: scanSlot?.id || '' });
        playAlertChime();
      }
    }, 3500);

    return () => window.clearInterval(timer);
  }, [cameraSlots, isScanning, language, role]);

  useEffect(() => {
    if (!isCameraReady) return;
    window.setTimeout(() => {
      void startVisibleCameras();
    }, 0);
  }, [cameraSlots.length]);

  async function refreshCameraList() {
    if (!navigator.mediaDevices?.enumerateDevices) {
      setCameraError('Camera devices are not available in this browser.');
      return [] as MediaDeviceInfo[];
    }

    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter((device) => device.kind === 'videoinput');
      setAvailableCameras(videoDevices);
      setCameraSlots((slots) =>
        slots.map((slot, index) => ({
          ...slot,
          deviceId: slot.deviceId || videoDevices[index]?.deviceId || videoDevices[0]?.deviceId || '',
        }))
      );
      setCameraError('');
      return videoDevices;
    } catch {
      setCameraError('Unable to read camera list. Please allow camera access.');
      return [] as MediaDeviceInfo[];
    }
  }

  function getCameraSlotLabel(slot: CameraSlot | undefined, index: number) {
    if (!slot) return language === 'BM' ? 'Kamera Laptop' : 'Laptop Camera';
    const device = availableCameras.find((cameraDevice) => cameraDevice.deviceId === slot.deviceId);
    return device?.label || (index === 0 ? (language === 'BM' ? 'Kamera Laptop' : 'Laptop Camera') : `Camera ${index + 1}`);
  }

  function stopCamera() {
    Object.values(activeStreamsRef.current).forEach((stream) => {
      stream.getTracks().forEach((track) => track.stop());
    });
    activeStreamsRef.current = {};
    setPreviewSlotIds([]);
    setIsCameraReady(false);
  }

  async function startCameraForSlot(slot: CameraSlot) {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError('Camera access is not supported in this browser.');
      return false;
    }

    try {
      const streamKey = slot.deviceId || 'default-camera';
      let stream = activeStreamsRef.current[streamKey];

      if (!stream) {
        stream = await navigator.mediaDevices.getUserMedia({
          video: slot.deviceId
            ? { deviceId: { exact: slot.deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
            : { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        activeStreamsRef.current[streamKey] = stream;
      }

      const video = videoRefs.current[slot.id];
      if (video) {
        video.muted = true;
        video.playsInline = true;
        video.setAttribute('playsinline', 'true');
        video.setAttribute('webkit-playsinline', 'true');
        video.srcObject = stream;
        await video.play().catch(() => undefined);
      }

      setPreviewSlotIds((ids) => (ids.includes(slot.id) ? ids : [...ids, slot.id]));
      setCameraError('');
      return true;
    } catch {
      setCameraError('Unable to start camera. Please allow camera permission and try again.');
      return false;
    }
  }

  async function startVisibleCameras() {
    stopCamera();
    const devices = await refreshCameraList();
    const slots =
      cameraSlots.length > 0
        ? cameraSlots.map((slot, index) => ({
            ...slot,
            deviceId: slot.deviceId || devices[index]?.deviceId || devices[0]?.deviceId || '',
          }))
        : [{ id: 'camera-slot-1', deviceId: devices[0]?.deviceId || '' }];

    setCameraSlots(slots);
    const startedResults = await Promise.all(slots.map((slot) => startCameraForSlot(slot)));
    const started = startedResults.some(Boolean);
    setIsCameraReady(started);
    return started;
  }

  const handleUseCamera = async (slot: CameraSlot, index: number) => {
    const devices = await refreshCameraList();
    const resolvedSlot = {
      ...slot,
      deviceId: slot.deviceId || devices[index]?.deviceId || devices[0]?.deviceId || '',
    };
    setCameraSlots((slots) => slots.map((item) => (item.id === slot.id ? resolvedSlot : item)));
    setActiveCameraSlotId(slot.id);
    const started = await startCameraForSlot(resolvedSlot);
    if (started) setIsCameraReady(true);
  };

  const handleCameraSlotDeviceChange = (slotId: string, deviceId: string) => {
    setCameraSlots((slots) => slots.map((slot) => (slot.id === slotId ? { ...slot, deviceId } : slot)));
    setPreviewSlotIds((ids) => ids.filter((id) => id !== slotId));
    setActiveCameraSlotId(slotId);
  };

  const handleStartScanning = async () => {
    const started = await startVisibleCameras();
    if (started) {
      setCurrentPlate('SCANNING');
      setIsScanning(true);
    }
  };

  const handlePauseScanning = () => {
    setIsScanning(false);
  };

  const handleAddCamera = async () => {
    const devices = await refreshCameraList();
    setCameraSlots((slots) => {
      if (slots.length >= 4) return slots;
      const nextIndex = slots.length;
      const nextDeviceId = devices[nextIndex]?.deviceId || devices[0]?.deviceId || slots[0]?.deviceId || '';
      return [...slots, { id: `camera-slot-${Date.now()}`, deviceId: nextDeviceId }];
    });
  };

  const handleRemoveCamera = (slotId: string) => {
    setCameraSlots((slots) => {
      if (slots.length <= 1) return slots;
      const nextSlots = slots.filter((slot) => slot.id !== slotId);
      if (activeCameraSlotId === slotId) {
        setActiveCameraSlotId(nextSlots[0]?.id || 'camera-slot-1');
      }
      setPreviewSlotIds((ids) => ids.filter((id) => id !== slotId));
      return nextSlots;
    });
  };

  function playAlertChime() {
    if (!soundEnabledRef.current) return;
    try {
      const audioWindow = window as AudioWindow;
      const AudioCtor = audioWindow.AudioContext || audioWindow.webkitAudioContext;
      if (!AudioCtor) return;
      const audioCtx = new AudioCtor();
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(880, audioCtx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(1760, audioCtx.currentTime + 0.3);
      gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.3);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + 0.3);
    } catch {
      // Silent fallback when audio playback is blocked by the browser.
    }
  }

  const handleMarkAsSeen = () => {
    if (activeAlertMatch) {
      const flaggedObj = { ...activeAlertMatch.vehicle, status: 'FLAGGED' as const };
      updateVehicle(flaggedObj);
      addHistoryLog({
        type: 'DETECTION',
        action: `Tanda Tindakan (Disemak): ${activeAlertMatch.vehicle.plate}`,
        plate: activeAlertMatch.vehicle.plate,
        details: `Alert Tanda Tindakan disemak oleh ${role} pada ${activeAlertMatch.cameraName}`,
        note: `Ditanda Tindakan oleh ${role} pada ${activeAlertMatch.cameraName}`,
        userRole: role,
        statusMatch: 'EXACT',
      });
    }
    setActiveAlertMatch(null);
  };

  return (
    <div className="space-y-4 max-w-6xl mx-auto px-1 sm:px-0">
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-3 sm:p-4 shadow-xl flex flex-col lg:flex-row items-stretch lg:items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <Link
            href="/"
            className="p-1.5 sm:p-2 rounded-xl bg-slate-800 text-slate-300 hover:bg-slate-700 hover:text-white transition-all shrink-0"
            title={language === 'BM' ? 'Kembali ke Papan Utama' : 'Back to Dashboard'}
          >
            <ArrowLeft className="w-4 h-4" />
          </Link>
          <div>
            <h1 className="text-xs sm:text-base font-black text-white uppercase tracking-wider">
              {t('liveScannerTitle')}
            </h1>
            <p className="text-[10px] text-slate-500 hidden sm:block">
              {isScanning ? 'Scanning live camera feed' : 'Choose camera in preview and start demo scan'}
            </p>
          </div>
        </div>

        <div className="flex items-center justify-end gap-2">
            <button
              type="button"
              onClick={() => void handleAddCamera()}
              disabled={cameraSlots.length >= 4}
              className="hidden sm:inline-flex items-center gap-1.5 rounded-xl border border-slate-700 bg-slate-800 px-3 py-2 text-xs font-bold text-slate-300 transition-all hover:border-cyan-700 hover:text-cyan-300 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <Plus className="w-4 h-4" />
              <span>Add Camera</span>
            </button>
            {isScanning ? (
              <button
                onClick={handlePauseScanning}
                className="flex-1 sm:flex-initial px-4 py-2 rounded-xl bg-amber-950 text-amber-300 border border-amber-700 text-xs font-black uppercase flex items-center justify-center gap-2"
              >
                <Pause className="w-4 h-4" />
                <span>Pause Scan</span>
              </button>
            ) : (
              <button
                onClick={() => void handleStartScanning()}
                className="flex-1 sm:flex-initial px-4 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-black uppercase flex items-center justify-center gap-2 shadow-lg shadow-cyan-500/20"
              >
                <Play className="w-4 h-4" />
                <span>Start Scanning</span>
              </button>
            )}
            <button
              onClick={() => setSoundEnabled(!soundEnabled)}
              className={`p-2 rounded-xl border text-xs font-bold transition-all shrink-0 ${
                soundEnabled
                  ? 'bg-cyan-950 text-cyan-400 border-cyan-800'
                  : 'bg-slate-800 text-slate-400 border-slate-700'
              }`}
              title="Toggle sound"
            >
              {soundEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
            </button>
        </div>
      </div>

      {cameraError && (
        <div className="rounded-xl border border-red-800 bg-red-950/40 px-4 py-3 text-xs font-bold text-red-300">
          {cameraError}
        </div>
      )}

      <div
        className="relative bg-slate-950 border border-slate-800 rounded-2xl overflow-hidden shadow-xl sm:aspect-video"
      >
        <div
          className={`grid gap-1.5 p-1.5 sm:absolute sm:inset-0 ${
            cameraSlots.length === 1
              ? 'grid-cols-1'
              : cameraSlots.length === 2
              ? 'grid-cols-1 sm:grid-cols-2'
              : 'grid-cols-1 sm:grid-cols-2'
          }`}
        >
          {cameraSlots.map((slot, index) => {
            const slotLabel = getCameraSlotLabel(slot, index);
            const isActiveSlot = activeCameraSlotId === slot.id;
            const isScanningSlot = isScanning && (lastDetectedSlotId ? lastDetectedSlotId === slot.id : isActiveSlot);
            const isAlertSlot = activeAlertMatch?.cameraId === slot.id;

            return (
              <div
                key={slot.id}
                onClick={() => setActiveCameraSlotId(slot.id)}
                className={`relative min-h-[420px] sm:min-h-0 rounded-xl overflow-hidden border bg-slate-950 text-left transition-all ${
                  isAlertSlot
                    ? 'border-red-500 ring-2 ring-red-500/40'
                    : isActiveSlot
                    ? 'border-cyan-500/70 ring-1 ring-cyan-500/30'
                    : 'border-slate-800'
                }`}
              >
                <video
                  ref={(node) => {
                    videoRefs.current[slot.id] = node;
                  }}
                  className="absolute inset-0 h-full w-full object-cover"
                  muted
                  playsInline
                  autoPlay
                />

                {!previewSlotIds.includes(slot.id) && (
                  <div className="absolute inset-0 bg-slate-950 flex flex-col items-center justify-center gap-3 text-center p-6">
                    <div className="w-12 h-12 rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-center">
                      <Video className="w-6 h-6 text-cyan-400" />
                    </div>
                    <div>
                      <div className="text-xs font-black text-white uppercase tracking-wider">{slotLabel}</div>
                      <div className="text-[10px] text-slate-500 mt-1">Choose camera and use preview before scanning.</div>
                    </div>
                  </div>
                )}

                {isScanningSlot && (
                  <>
                    <div className="absolute inset-0 bg-[linear-gradient(rgba(6,182,212,0.10)_1px,transparent_1px),linear-gradient(90deg,rgba(6,182,212,0.10)_1px,transparent_1px)] bg-[size:48px_48px]" />
                    <div className="absolute inset-x-0 h-0.5 bg-cyan-400/90 animate-laser shadow-[0_0_16px_#06B6D4]" />
                    <div className="absolute inset-4 border border-cyan-400/40 rounded-lg shadow-[0_0_30px_rgba(6,182,212,0.20)_inset]" />
                  </>
                )}

                <div className="absolute inset-x-2.5 top-2.5 z-20 flex items-center gap-1.5">
                  <span className={`h-2 w-2 rounded-full shrink-0 ${isAlertSlot ? 'bg-red-400 animate-pulse' : isScanningSlot ? 'bg-emerald-400 animate-pulse' : previewSlotIds.includes(slot.id) ? 'bg-cyan-400' : 'bg-slate-500'}`} />
                  <select
                    value={slot.deviceId}
                    onClick={(event) => event.stopPropagation()}
                    onChange={(event) => handleCameraSlotDeviceChange(slot.id, event.target.value)}
                    className="h-7 min-w-0 flex-1 rounded-lg border border-slate-700 bg-slate-900/90 px-2 text-[10px] font-bold text-slate-200 outline-none hover:border-cyan-700 focus:border-cyan-500"
                    title={language === 'BM' ? 'Pilih kamera' : 'Choose camera'}
                  >
                    {(availableCameras.length > 0 ? availableCameras : [{ deviceId: '', label: slotLabel } as MediaDeviceInfo]).map(
                      (cameraDevice, cameraIndex) => (
                        <option key={cameraDevice.deviceId || `camera-${cameraIndex}`} value={cameraDevice.deviceId}>
                          {cameraDevice.label || (cameraIndex === 0 ? slotLabel : `Camera ${cameraIndex + 1}`)}
                        </option>
                      )
                    )}
                  </select>
                  <button
                    type="button"
                    onClick={(event) => {
                      event.stopPropagation();
                      void handleUseCamera(slot, index);
                    }}
                    className="h-7 rounded-lg border border-cyan-700 bg-cyan-950/90 px-2 text-[10px] font-bold text-cyan-200 hover:bg-cyan-900"
                  >
                    Preview
                  </button>
                  {cameraSlots.length > 1 && (
                    <button
                      type="button"
                      onClick={(event) => {
                        event.stopPropagation();
                        handleRemoveCamera(slot.id);
                      }}
                      className="h-7 w-7 rounded-lg border border-slate-700 bg-slate-900/90 text-slate-300 hover:border-red-700 hover:text-red-300"
                      title={language === 'BM' ? 'Buang kamera' : 'Remove camera'}
                    >
                      <X className="mx-auto h-3.5 w-3.5" />
                    </button>
                  )}
                </div>

                {isActiveSlot && (
                  <div className="absolute bottom-2.5 left-2.5 bg-cyan-950/90 text-cyan-200 border border-cyan-500/60 rounded-lg px-2.5 py-1 text-[10px] font-bold">
                    Active scanner
                  </div>
                )}
                {isScanningSlot && (
                  <div className={`absolute bottom-2.5 ${isActiveSlot ? 'left-32' : 'left-2.5'} bg-slate-900/90 text-cyan-200 border border-cyan-700 rounded-lg px-2.5 py-1 text-[10px] font-mono font-black`}>
                    {currentPlate}
                  </div>
                )}
                <button
                  type="button"
                  onClick={(event) => {
                    event.stopPropagation();
                    void handleAddCamera();
                  }}
                  disabled={cameraSlots.length >= 4}
                  className="sm:hidden absolute bottom-2.5 right-2.5 z-20 inline-flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-900/85 px-2.5 py-1 text-[10px] font-bold text-slate-300 disabled:opacity-40"
                >
                  <Plus className="h-3 w-3" />
                  <span>Add</span>
                </button>
                {isAlertSlot && activeAlertMatch && (
                  <div className="scanner-alert-card absolute inset-1.5 z-30 bg-red-600 border-2 border-red-400 rounded-xl p-2.5 sm:p-4 shadow-2xl flex flex-col justify-between text-center overflow-hidden">
                    <div className="flex items-center justify-between border-b border-red-500/80 pb-2">
                      <div className="flex items-center gap-2 min-w-0">
                        <ShieldAlert className="w-4 h-4 text-white animate-bounce shrink-0" />
                        <span className="text-[10px] sm:text-xs font-black uppercase text-white tracking-wider truncate">
                          {language === 'BM' ? 'AMARAN: PADANAN DIKESAN' : 'ALERT: MATCH DETECTED'}
                        </span>
                      </div>
                      <button
                        onClick={(event) => {
                          event.stopPropagation();
                          setActiveAlertMatch(null);
                        }}
                        className="p-1 rounded-lg bg-red-700 hover:bg-red-800 text-white transition-all"
                        title={t('closeBtn')}
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>

                    <div className="bg-red-700 border border-red-400/50 rounded-lg p-2.5 sm:p-3 space-y-2 text-left shadow-inner">
                      <div className="grid grid-cols-2 gap-2 border-b border-red-500/80 pb-2">
                        <div>
                          <span className="text-[9px] text-red-100 font-bold uppercase block">{t('plateNumber')}</span>
                          <span className="plate-yellow text-base sm:text-xl font-mono font-black">
                            {activeAlertMatch.vehicle.plate}
                          </span>
                        </div>
                        <div>
                          <span className="text-[9px] text-red-100 font-bold uppercase block">{t('outstandingAmount')}</span>
                          <span className="text-base sm:text-lg font-mono font-black text-white">
                            {formatMYR(activeAlertMatch.vehicle.outstandingAmount)}
                          </span>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-[10px] sm:text-xs">
                        <div>
                          <span className="text-red-100 block">{t('vehicleDetails')}</span>
                          <strong className="text-white truncate block">
                            {activeAlertMatch.vehicle.brand} {activeAlertMatch.vehicle.model}
                          </strong>
                        </div>
                        <div>
                          <span className="text-red-100 block">{language === 'BM' ? 'Kamera' : 'Camera'}</span>
                          <strong className="text-white truncate block">{activeAlertMatch.cameraName}</strong>
                        </div>
                      </div>
                    </div>

                    <div className="flex flex-wrap items-center justify-center gap-1.5 pt-2 border-t border-red-500/80">
                      <button
                        onClick={(event) => {
                          event.stopPropagation();
                          handleMarkAsSeen();
                        }}
                        className="btn-mark-seen px-3 py-1.5 rounded-lg bg-white hover:bg-slate-100 text-red-600 text-[10px] sm:text-xs font-black uppercase tracking-wider flex items-center justify-center gap-1 shadow-md transition-all"
                      >
                        <BookmarkCheck className="w-3.5 h-3.5 text-red-600" />
                        <span>{t('markAction')}</span>
                      </button>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-3.5 sm:p-5 shadow-xl space-y-3">
        <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400" />
            <h2 className="text-xs sm:text-sm font-bold text-white uppercase tracking-wider">
              {t('recentDetectionList')}
            </h2>
          </div>
          <span className="text-[10px] font-mono text-slate-500">
            {liveDetections.length} session scans
          </span>
        </div>

        <div className="sm:hidden space-y-2">
          {liveDetections.length > 0 ? (
            liveDetections.map((det) => (
              <button
                key={det.id}
                type="button"
                onClick={() => setExpandedDetectionId((current) => (current === det.id ? null : det.id))}
                className="w-full p-2.5 rounded-xl bg-slate-950/80 border border-slate-800 text-left space-y-2 hover:border-cyan-900/70 transition-all"
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="space-y-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-mono font-black text-cyan-300 text-xs">{det.plate}</span>
                      <span className="text-[10px] text-slate-400 truncate">{det.cameraName}</span>
                    </div>
                    {det.matched ? (
                      <span className="px-1.5 py-0.5 rounded-full text-[9px] font-bold uppercase bg-red-950 text-red-400 border border-red-700 inline-flex items-center gap-1">
                        <CheckCircle2 className="w-2.5 h-2.5" />
                        <span>TANDA TINDAKAN</span>
                      </span>
                    ) : (
                      <span className="px-1.5 py-0.5 rounded-full text-[9px] font-bold uppercase bg-slate-900 text-slate-400 border border-slate-800 inline-flex items-center gap-1">
                        <XCircle className="w-2.5 h-2.5" />
                        <span>{language === 'BM' ? 'TIADA' : 'NO MATCH'}</span>
                      </span>
                    )}
                  </div>
                  <span className="text-[10px] font-mono text-slate-500 shrink-0">{det.timestamp}</span>
                </div>
                {expandedDetectionId === det.id && (
                  <div className="grid grid-cols-1 gap-1 border-t border-slate-800 pt-2 text-[10px] text-slate-400">
                    <div className="flex items-center gap-1.5">
                      <MapPin className="h-3 w-3 text-cyan-400" />
                      <span>{det.name}</span>
                    </div>
                    <div className="font-mono">GPS: {det.gps}</div>
                  </div>
                )}
              </button>
            ))
          ) : (
            <div className="py-5 text-center text-xs text-slate-500">No scans in this session yet.</div>
          )}
        </div>

        <div className="hidden sm:block overflow-x-auto">
          <table className="w-full text-left text-xs min-w-[560px]">
            <thead className="bg-slate-950 text-slate-400 uppercase font-mono text-[9px] sm:text-[10px] border-b border-slate-800 whitespace-nowrap">
              <tr>
                <th className="py-2 px-3">{language === 'BM' ? 'Nombor Plat' : 'Plate Number'}</th>
                <th className="py-2 px-3">Status</th>
                <th className="py-2 px-3">Confidence</th>
                <th className="py-2 px-3">{language === 'BM' ? 'Kamera' : 'Camera'}</th>
                <th className="py-2 px-3 text-right">{language === 'BM' ? 'Masa' : 'Time'}</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono whitespace-nowrap">
              {liveDetections.length > 0 ? (
                liveDetections.map((det) => (
                  <React.Fragment key={det.id}>
                    <tr
                      onClick={() => setExpandedDetectionId((current) => (current === det.id ? null : det.id))}
                      className="hover:bg-slate-800/40 transition-colors cursor-pointer"
                    >
                      <td className="py-2 px-3 font-black text-cyan-300 text-xs sm:text-sm">{det.plate}</td>
                      <td className="py-2 px-3">
                        {det.matched ? (
                          <span className="px-2 py-0.5 rounded-full text-[9px] sm:text-[10px] font-bold uppercase bg-red-950 text-red-400 border border-red-700 inline-flex items-center gap-1">
                            <CheckCircle2 className="w-3 h-3" />
                            <span>TANDA TINDAKAN</span>
                          </span>
                        ) : (
                          <span className="px-2 py-0.5 rounded-full text-[9px] sm:text-[10px] font-bold uppercase bg-slate-950 text-slate-400 border border-slate-800 inline-flex items-center gap-1">
                            <XCircle className="w-3 h-3" />
                            <span>{language === 'BM' ? 'TIADA PADANAN' : 'UNMATCHED'}</span>
                          </span>
                        )}
                      </td>
                      <td className="py-2 px-3 text-slate-300 text-[11px]">{det.confidence}%</td>
                      <td className="py-2 px-3 text-slate-300 text-[11px]">{det.cameraName}</td>
                      <td className="py-2 px-3 text-right text-slate-500 text-[10px] sm:text-[11px]">{det.timestamp}</td>
                    </tr>
                    {expandedDetectionId === det.id && (
                      <tr className="bg-slate-950/70">
                        <td colSpan={5} className="px-3 py-3">
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-2 text-[11px] text-slate-400">
                            <div className="flex items-center gap-1.5 md:col-span-2">
                              <MapPin className="h-3.5 w-3.5 text-cyan-400" />
                              <span className="text-slate-500">{language === 'BM' ? 'Lokasi Dijumpai:' : 'Found Location:'}</span>
                              <span className="text-slate-300">{det.name}</span>
                            </div>
                            <div className="font-mono text-slate-300">GPS: {det.gps}</div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))
              ) : (
                <tr>
                  <td colSpan={5} className="py-8 text-center text-slate-500 font-sans">
                    No scans in this session yet. Start scanning to populate dashboard and audit history.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
