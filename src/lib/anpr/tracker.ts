import { TrackOcrState } from '../db/types';

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
}

export type TrackLifecycleState = 'VISIBLE' | 'LOST' | 'REMOVED';
export type PlatePipelineState =
  | 'DETECTED'
  | 'TRACKING'
  | 'COLLECTING'
  | 'READY_FOR_OCR'
  | 'READING'
  | 'CONSENSUS'
  | 'MATCHED'
  | 'COOLDOWN'
  | 'FINISHED';

export interface TrackConfidenceComponents {
  detection: number;
  age: number;
  motion: number;
  iou: number;
  ocr: number;
}

export interface TrackRuntimeStats {
  startedAt: number;
  lastUpdatedAt: number;
  framesVisible: number;
  framesLost: number;
  ocrAttempts: number;
  ocrAccepted: number;
  consensusAttempts: number;
  finalConfidence?: number;
  finishedAt?: number;
  lastDetectorLatencyMs?: number;
  lastOcrLatencyMs?: number;
  bestCropQuality?: number;
}

export interface TrackCropSample {
  dataUrl?: string;
  qualityScore: number;
  timestamp: number;
  ocrText?: string;
  ocrConfidence?: number;
}

export interface ActiveTrack {
  trackId: string;
  trackNumber: number;
  bbox: BoundingBox;          // Raw detection bbox (used for IoU / matching / cropping)
  smoothBbox: BoundingBox;    // EMA-smoothed bbox (used strictly for UI overlay rendering)
  predictedBbox?: BoundingBox;
  overlayAngle?: number;
  lastOverlayAngleAt?: number;
  vx: number; // velocity x (pixels per frame)
  vy: number; // velocity y (pixels per frame)
  cropSamples: TrackCropSample[];
  lastSeenFrame: number;
  firstSeenFrame: number;
  lastSeenTimestamp: number;
  firstSeenTimestamp: number;
  framesSeen: number;
  visibleThisFrame?: boolean;
  missedFrames?: number;
  motionScore?: number;
  trackConfidence?: number;
  confidenceComponents?: TrackConfidenceComponents;
  trackState?: TrackLifecycleState;
  pipelineState?: PlatePipelineState;
  stats?: TrackRuntimeStats;

  ocrState: TrackOcrState;
  ocrRunning: boolean;
  ocrJobQueued: boolean;
  lastCropSampledAt?: number;
  lastOcrAttemptAt?: number;
  lastOcrCompletedAt?: number;

  votes: Map<string, { count: number; totalConfidence: number }>;
  stabilizedPlate?: string;
  stabilizedConfidence?: number;

  matchType?: 'EXACT' | 'POSSIBLE' | 'NONE';
  matchedVehicle?: any;
  possibleMatchVehicles?: any[];

  possibleVerificationPlate?: string;
  possibleVerificationCount?: number;
  possibleVerificationStartedAt?: number;

  cooldownActive: boolean;
  cooldownStartedAt?: number;
  lastSearchedAt?: number;
  scanEventId?: string;

  isConfirmed?: boolean;
}

export function calculateIoU(boxA: BoundingBox, boxB: BoundingBox): number {
  const xA = Math.max(boxA.x, boxB.x);
  const yA = Math.max(boxA.y, boxB.y);
  const xB = Math.min(boxA.x + boxA.width, boxB.x + boxB.width);
  const yB = Math.min(boxA.y + boxA.height, boxB.y + boxB.height);

  const interWidth = Math.max(0, xB - xA);
  const interHeight = Math.max(0, yB - yA);
  const interArea = interWidth * interHeight;

  if (interArea === 0) return 0;

  const boxAArea = boxA.width * boxA.height;
  const boxBArea = boxB.width * boxB.height;

  return interArea / (boxAArea + boxBArea - interArea);
}

export function calculateCentroidDistance(boxA: BoundingBox, boxB: BoundingBox): number {
  const cxA = boxA.x + boxA.width / 2;
  const cyA = boxA.y + boxA.height / 2;
  const cxB = boxB.x + boxB.width / 2;
  const cyB = boxB.y + boxB.height / 2;
  return Math.sqrt(Math.pow(cxA - cxB, 2) + Math.pow(cyA - cyB, 2));
}

/**
 * ByteTrack-inspired Multi-Object Tracker for Real-Time License Plate Tracking.
 * 
 * Key Features:
 * - High-confidence & low-confidence two-stage association.
 * - Velocity prediction (dx, dy) for moving vehicles & moving cameras.
 * - Max active tracks pool (default 8).
 * - Independent track memory buffer & state machine.
 */
export class PlateTracker {
  private activeTracks: Map<string, ActiveTrack> = new Map();
  private trackCounter: number = 1;
  private frameIndex: number = 0;
  private iouThreshold: number = 0.30;
  private lostTrackTimeout: number = 20; // Frames before a confirmed track is pruned (~2s at 10 FPS)
  private lostTrackTimeoutMs: number = 2000; // Time-based expiration in milliseconds (invariant to FPS drops)
  private completedTrackLostTimeoutMs: number = 800;
  private maxActiveTracks: number = 8;
  private minConfirmationFrames: number = 2;

  private isScaleCompatible(reference: BoundingBox, candidate: BoundingBox): boolean {
    const widthRatio = candidate.width / Math.max(1, reference.width);
    const heightRatio = candidate.height / Math.max(1, reference.height);
    const referenceAspect = reference.width / Math.max(1, reference.height);
    const candidateAspect = candidate.width / Math.max(1, candidate.height);
    const aspectRatio = candidateAspect / Math.max(0.1, referenceAspect);

    return (
      widthRatio >= 0.45 &&
      widthRatio <= 2.20 &&
      heightRatio >= 0.45 &&
      heightRatio <= 2.20 &&
      aspectRatio >= 0.45 &&
      aspectRatio <= 2.20
    );
  }

  private getAssociationDistanceLimit(track: ActiveTrack, targetBox: BoundingBox, confidence: number): number {
    const baseSize = Math.max(targetBox.width, targetBox.height, 1);
    const framesMissing = Math.max(0, this.frameIndex - track.lastSeenFrame - 1);
    const velocityBoost = Math.min(baseSize * 0.75, Math.hypot(track.vx, track.vy) * Math.max(1, framesMissing + 1));
    const missedFrameBoost = Math.min(baseSize * 0.45, framesMissing * baseSize * 0.15);
    const confidenceBoost = confidence >= 0.70 ? baseSize * 0.15 : 0;

    return baseSize * 1.65 + velocityBoost + missedFrameBoost + confidenceBoost;
  }

  private shouldSkipAssociationAfterGap(track: ActiveTrack): boolean {
    const frameGap = this.frameIndex - track.lastSeenFrame;
    const hasOcrMemory = track.cooldownActive || Boolean(track.stabilizedPlate) || track.votes.size > 0;

    if (track.cooldownActive && frameGap > 1) return true;
    if (hasOcrMemory && frameGap > 1) return true;

    return false;
  }

  private getOcrStabilityScore(track: ActiveTrack): number {
    let totalVotes = 0;
    let topVotes = 0;

    track.votes.forEach((vote) => {
      totalVotes += vote.count;
      topVotes = Math.max(topVotes, vote.count);
    });

    if (totalVotes === 0) return 0.55;
    return topVotes / totalVotes;
  }

  private calculateTrackConfidence(
    track: ActiveTrack,
    detectorConfidence: number,
    associationIoU: number
  ): number {
    const components: TrackConfidenceComponents = {
      detection: detectorConfidence,
      age: Math.min(1, track.framesSeen / 4),
      motion: 1 - Math.min(1, (track.motionScore ?? 0) / 0.50),
      iou: associationIoU > 0 ? Math.min(1, associationIoU / 0.50) : 0.55,
      ocr: this.getOcrStabilityScore(track),
    };

    const confidence =
      components.detection * 0.38 +
      components.age * 0.18 +
      components.motion * 0.16 +
      components.iou * 0.16 +
      components.ocr * 0.12;

    track.confidenceComponents = components;
    return Math.round(Math.min(1, Math.max(0, confidence)) * 100) / 100;
  }

  private calculateInitialTrackConfidence(detectorConfidence: number): number {
    const components: TrackConfidenceComponents = {
      detection: detectorConfidence,
      age: 0.25,
      motion: 1.0,
      iou: 0.55,
      ocr: 0.55,
    };

    const confidence =
      components.detection * 0.38 +
      components.age * 0.18 +
      components.motion * 0.16 +
      components.iou * 0.16 +
      components.ocr * 0.12;

    return Math.round(Math.min(1, Math.max(0, confidence)) * 100) / 100;
  }

  private createInitialConfidenceComponents(detectorConfidence: number): TrackConfidenceComponents {
    return {
      detection: detectorConfidence,
      age: 0.25,
      motion: 1,
      iou: 0.55,
      ocr: 0.55,
    };
  }

  private markMatchedTrack(
    track: ActiveTrack,
    matchedBox: BoundingBox,
    velocityAlpha: number,
    smoothAlpha: number,
    associationIoU: number
  ): void {
    const dx = matchedBox.x - track.bbox.x;
    const dy = matchedBox.y - track.bbox.y;
    const motionDenominator = Math.max(1, matchedBox.width, matchedBox.height);

    track.vx = track.vx * (1 - velocityAlpha) + dx * velocityAlpha;
    track.vy = track.vy * (1 - velocityAlpha) + dy * velocityAlpha;
    track.motionScore = Math.min(2, Math.hypot(dx, dy) / motionDenominator);

    const now = Date.now();
    track.bbox = matchedBox;

    // EMA smoothing is only for UI display, so raw crops still use the detector bbox.
    track.smoothBbox = {
      x: track.smoothBbox.x * (1 - smoothAlpha) + matchedBox.x * smoothAlpha,
      y: track.smoothBbox.y * (1 - smoothAlpha) + matchedBox.y * smoothAlpha,
      width: track.smoothBbox.width * (1 - smoothAlpha) + matchedBox.width * smoothAlpha,
      height: track.smoothBbox.height * (1 - smoothAlpha) + matchedBox.height * smoothAlpha,
      confidence: matchedBox.confidence,
    };

    track.lastSeenFrame = this.frameIndex;
    track.lastSeenTimestamp = now;
    track.framesSeen++;
    track.visibleThisFrame = true;
    track.missedFrames = 0;
    track.trackState = 'VISIBLE';
    if (track.pipelineState === 'DETECTED') track.pipelineState = 'TRACKING';
    track.trackConfidence = this.calculateTrackConfidence(track, matchedBox.confidence, associationIoU);
    if (track.stats) {
      track.stats.framesVisible++;
      track.stats.lastUpdatedAt = now;
    }
    if (track.framesSeen >= this.minConfirmationFrames || track.bbox.confidence >= 0.70) track.isConfirmed = true;
  }

  constructor(lostTrackTimeout?: number, maxActiveTracks?: number, minConfirmationFrames?: number) {
    if (lostTrackTimeout !== undefined) {
      this.lostTrackTimeout = lostTrackTimeout;
      this.lostTrackTimeoutMs = lostTrackTimeout * 100;
      this.completedTrackLostTimeoutMs = Math.min(800, this.lostTrackTimeoutMs);
    }
    if (maxActiveTracks !== undefined) this.maxActiveTracks = maxActiveTracks;
    if (minConfirmationFrames !== undefined) this.minConfirmationFrames = minConfirmationFrames;
  }

  public updateTracks(detectedBoxes: BoundingBox[]): ActiveTrack[] {
    this.frameIndex++;

    this.activeTracks.forEach((track) => {
      track.visibleThisFrame = false;
      track.missedFrames = Math.max(0, this.frameIndex - track.lastSeenFrame);
      track.trackState = 'LOST';
    });

    // 1. Separate detections into High Confidence and Low Confidence
    const highConfDets: { box: BoundingBox; idx: number }[] = [];
    const lowConfDets: { box: BoundingBox; idx: number }[] = [];

    detectedBoxes.forEach((box, idx) => {
      if (box.confidence >= 0.40) {
        highConfDets.push({ box, idx });
      } else {
        lowConfDets.push({ box, idx });
      }
    });

    const unassignedHigh = new Set<number>(highConfDets.map(d => d.idx));
    const unassignedLow = new Set<number>(lowConfDets.map(d => d.idx));

    // 2. Predict next position for active tracks using velocity (Kalman/Constant Velocity model)
    this.activeTracks.forEach((track) => {
      const dt = this.frameIndex - track.lastSeenFrame;
      track.predictedBbox = {
        x: track.bbox.x + track.vx * dt,
        y: track.bbox.y + track.vy * dt,
        width: track.bbox.width,
        height: track.bbox.height,
        confidence: track.bbox.confidence,
      };
    });

    // 3. First Stage Association: Match Active Tracks with High Confidence Detections
    this.activeTracks.forEach((track) => {
      if (this.shouldSkipAssociationAfterGap(track)) return;

      let bestMatchScore = 0;
      let bestIdx = -1;
      let bestIoU = 0;

      highConfDets.forEach(({ box, idx }) => {
        if (!unassignedHigh.has(idx)) return;
        const targetBox = track.predictedBbox || track.bbox;
        if (!this.isScaleCompatible(targetBox, box)) return;

        const iou = calculateIoU(targetBox, box);
        const dist = calculateCentroidDistance(targetBox, box);
        const maxDist = this.getAssociationDistanceLimit(track, targetBox, box.confidence);

        let score = 0;
        if (iou > this.iouThreshold) {
          score = 1.0 + iou; // prioritize IoU
        } else if (dist < maxDist) {
          score = 1.0 - (dist / maxDist); // fallback to distance
        }

        if (score > bestMatchScore) {
          bestMatchScore = score;
          bestIdx = idx;
          bestIoU = iou;
        }
      });

      if (bestIdx !== -1) {
        const matchedBox = detectedBoxes[bestIdx];
        this.markMatchedTrack(track, matchedBox, 0.30, 0.35, bestIoU);
        unassignedHigh.delete(bestIdx);
      }
    });

    // 4. Second Stage Association: Match Unassigned Tracks with Low Confidence Detections
    this.activeTracks.forEach((track) => {
      if (track.lastSeenFrame === this.frameIndex) return; // Already updated in Stage 1
      if (this.shouldSkipAssociationAfterGap(track)) return;

      let bestMatchScore = 0;
      let bestIdx = -1;
      let bestIoU = 0;

      lowConfDets.forEach(({ box, idx }) => {
        if (!unassignedLow.has(idx)) return;
        const targetBox = track.predictedBbox || track.bbox;
        if (!this.isScaleCompatible(targetBox, box)) return;

        const iou = calculateIoU(targetBox, box);
        const dist = calculateCentroidDistance(targetBox, box);
        const maxDist = this.getAssociationDistanceLimit(track, targetBox, box.confidence) * 0.85;

        let score = 0;
        if (iou > this.iouThreshold * 0.8) {
          score = 1.0 + iou;
        } else if (dist < maxDist) {
          score = 1.0 - (dist / maxDist);
        }

        if (score > bestMatchScore) {
          bestMatchScore = score;
          bestIdx = idx;
          bestIoU = iou;
        }
      });

      if (bestIdx !== -1) {
        const matchedBox = detectedBoxes[bestIdx];
        this.markMatchedTrack(track, matchedBox, 0.25, 0.25, bestIoU);
        unassignedLow.delete(bestIdx);
      }
    });

    // 5. Create New Tracks for Unassigned High Confidence Detections
    unassignedHigh.forEach((idx) => {
      // Enforce max active tracks limit
      if (this.activeTracks.size >= this.maxActiveTracks) return;

      const box = detectedBoxes[idx];
      if (box.width < 35 || box.height < 10) return;

      const num = this.trackCounter++;
      const now = Date.now();
      const newTrack: ActiveTrack = {
        trackId: `TRK-${num}`,
        trackNumber: num,
        bbox: { ...box },
        smoothBbox: { ...box },
        vx: 0,
        vy: 0,
        cropSamples: [],
        lastSeenFrame: this.frameIndex,
        firstSeenFrame: this.frameIndex,
        lastSeenTimestamp: now,
        firstSeenTimestamp: now,
        framesSeen: 1,
        visibleThisFrame: true,
        missedFrames: 0,
        motionScore: 0,
        trackConfidence: this.calculateInitialTrackConfidence(box.confidence),
        confidenceComponents: this.createInitialConfidenceComponents(box.confidence),
        trackState: 'VISIBLE',
        pipelineState: 'DETECTED',
        stats: {
          startedAt: now,
          lastUpdatedAt: now,
          framesVisible: 1,
          framesLost: 0,
          ocrAttempts: 0,
          ocrAccepted: 0,
          consensusAttempts: 0,
        },
        ocrState: 'DETECTED',
        ocrRunning: false,
        ocrJobQueued: false,
        votes: new Map(),
        cooldownActive: false,
        isConfirmed: box.confidence >= 0.55, // Instantly confirm confident YOLO plates for fast-moving scenes
      };
      this.activeTracks.set(newTrack.trackId, newTrack);
    });

    // 6. Remove Stale Tracks by Timestamp (2000 ms timeout for confirmed tracks)
    const now = Date.now();
    this.activeTracks.forEach((track, id) => {
      const timeLostMs = now - (track.lastSeenTimestamp || 0);
      const timeoutMs = track.isConfirmed
        ? track.cooldownActive
          ? this.completedTrackLostTimeoutMs
          : this.lostTrackTimeoutMs
        : 600; // Unconfirmed expire quickly (600ms)
      if (timeLostMs > timeoutMs) {
        track.trackState = 'REMOVED';
        track.pipelineState = track.cooldownActive ? 'FINISHED' : track.pipelineState;
        if (track.stats) {
          track.stats.finishedAt = now;
          track.stats.finalConfidence = track.stabilizedConfidence ?? track.trackConfidence;
          track.stats.lastUpdatedAt = now;
        }
        this.activeTracks.delete(id);
      } else if (!track.visibleThisFrame) {
        track.missedFrames = Math.max(1, this.frameIndex - track.lastSeenFrame);
        track.trackState = 'LOST';
        track.trackConfidence = Math.round(Math.max(0, (track.trackConfidence ?? 0) * 0.65) * 100) / 100;
        if (track.stats) {
          track.stats.framesLost++;
          track.stats.lastUpdatedAt = now;
        }
      }
    });

    return Array.from(this.activeTracks.values());
  }

  public getActiveTracks(confirmedOnly: boolean = false): ActiveTrack[] {
    const all = Array.from(this.activeTracks.values());
    if (confirmedOnly) return all.filter(t => t.isConfirmed);
    return all;
  }

  public getTrack(trackId: string): ActiveTrack | undefined {
    return this.activeTracks.get(trackId);
  }

  public setLostTrackTimeout(frames: number): void {
    this.lostTrackTimeout = frames;
    this.lostTrackTimeoutMs = frames * 100;
    this.completedTrackLostTimeoutMs = Math.min(800, this.lostTrackTimeoutMs);
  }

  public setMaxActiveTracks(maxTracks: number): void {
    if (Number.isFinite(maxTracks) && maxTracks > 0) {
      this.maxActiveTracks = Math.max(1, Math.floor(maxTracks));
    }
  }

  public clear(): void {
    this.activeTracks.clear();
    this.frameIndex = 0;
    this.trackCounter = 1;
  }
}
