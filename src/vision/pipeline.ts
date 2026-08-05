/**
 * Turning a pair of frames into a scored dart.
 */

import { RADIUS, scoreFromCartesian, type BoardScore } from '../game/board'
import { imagePointToBoard, type Calibration } from './calibration'
import type { VisionConfig } from './config'
import {
  changedFraction,
  close,
  differenceMask,
  maskWhere,
  open,
  type GrayImage,
  type Mask,
} from './frameDiff'
import { axisExtremes, blobShape, findBlobs, type Blob } from './blobs'
import type { Point } from './homography'

export interface DartReading {
  score: BoardScore
  /** Where the tip was found, in frame pixels. */
  imagePoint: Point
  /** Where that maps to on the board, in millimetres. */
  boardPoint: Point
  /** `[0, 1]`. Below the configured threshold the reading is offered, not taken. */
  confidence: number
  blob: Blob
  /** Retained for the debug overlay. */
  mask: Mask
}

export type DetectionFailure =
  | 'no-change'
  | 'still-moving'
  | 'nothing-on-board'
  | 'too-small'
  | 'too-large'
  | 'not-dart-shaped'

export type DetectionResult =
  | { ok: true; reading: DartReading }
  | { ok: false; reason: DetectionFailure; mask: Mask | null }

/** How far outside the double ring a blob centroid may sit and still count. */
const BOARD_MARGIN = 1.15

/**
 * Find the dart that appeared between two settled frames.
 *
 * Pure: same inputs, same answer, no camera involved — which is what lets the
 * whole thing be tested against generated frames.
 */
export function detectDart(
  reference: GrayImage,
  current: GrayImage,
  calibration: Calibration,
  config: VisionConfig,
): DetectionResult {
  const raw = differenceMask(reference, current, config.pixelThreshold)
  if (raw.count === 0) return { ok: false, reason: 'no-change', mask: raw }

  // Open removes sensor speckle; close reconnects a dart broken up by a
  // highlight running along the barrel.
  const cleaned = close(open(raw))
  if (cleaned.count === 0) return { ok: false, reason: 'no-change', mask: cleaned }

  const onBoard = maskWhere(cleaned, (x, y) => {
    const board = imagePointToBoard(calibration, { x, y })
    return Math.hypot(board.x, board.y) <= RADIUS.doubleOuter * BOARD_MARGIN
  })

  if (onBoard.count / cleaned.count < config.minOnBoardFraction) {
    return { ok: false, reason: 'nothing-on-board', mask: cleaned }
  }

  const blobs = findBlobs(onBoard, 1)
  const largest = blobs[0]
  if (!largest) return { ok: false, reason: 'no-change', mask: onBoard }
  if (largest.area < config.minBlobArea) return { ok: false, reason: 'too-small', mask: onBoard }
  if (largest.area > config.maxBlobArea) return { ok: false, reason: 'too-large', mask: onBoard }

  const shape = blobShape(largest)
  if (shape.compactness > config.maxCompactness) {
    return { ok: false, reason: 'not-dart-shaped', mask: onBoard }
  }

  const [low, high] = axisExtremes(largest, shape)

  // Which end is the tip? The flight is nearer the camera than the tip, and a
  // lens magnifies what is closer, so the flight projects *further* from the
  // board centre than the point the dart actually entered. Taking the end
  // closer to the bull therefore picks the tip.
  //
  // This is also where a single camera reaches its limit: the reasoning holds
  // for a phone mounted square-on at board height, and degrades as the camera
  // moves off that axis. No amount of care here recovers the depth a second
  // camera would give.
  const lowBoard = imagePointToBoard(calibration, low)
  const highBoard = imagePointToBoard(calibration, high)
  const lowRadius = Math.hypot(lowBoard.x, lowBoard.y)
  const highRadius = Math.hypot(highBoard.x, highBoard.y)
  const [imagePoint, boardPoint] =
    lowRadius <= highRadius ? [low, lowBoard] : [high, highBoard]

  const score = scoreFromCartesian(boardPoint.x, boardPoint.y)

  return {
    ok: true,
    reading: {
      score,
      imagePoint,
      boardPoint,
      confidence: confidenceOf(largest, shape.compactness, onBoard, cleaned),
      blob: largest,
      mask: onBoard,
    },
  }
}

/**
 * How much to trust a reading, in `[0, 1]`.
 *
 * Three things go into it: the change was one clean object rather than several
 * (a second blob usually means a dart was knocked or a shadow moved), the
 * blob is elongated like a dart rather than round like a hand, and it sits in
 * the size range a dart occupies at this resolution.
 */
function confidenceOf(blob: Blob, compactness: number, onBoard: Mask, cleaned: Mask): number {
  const dominance = onBoard.count > 0 ? blob.area / onBoard.count : 0
  const elongation = 1 - Math.min(1, compactness / 0.9)
  const tidiness = cleaned.count > 0 ? onBoard.count / cleaned.count : 0

  const score = 0.45 * dominance + 0.35 * elongation + 0.2 * tidiness
  return Math.max(0, Math.min(1, score))
}

export type WatcherPhase = 'waiting' | 'moving' | 'settling'

export interface WatcherUpdate {
  phase: WatcherPhase
  /** Set on the frame a dart is recognised. */
  reading?: DartReading
  /** Why a settled frame produced nothing, for the debug overlay. */
  failure?: DetectionFailure
  /** Fraction of the frame currently changing. */
  motion: number
}

/**
 * Watches a stream of frames and reports darts as they land.
 *
 * The sequence is always the same: the scene is still, then something moves
 * (an arm, the dart in flight), then it goes still again — and whatever is
 * different now is the dart. Waiting for stillness before reading is what
 * keeps hands and shadows out of the scoring.
 */
export class ThrowWatcher {
  private reference: GrayImage | null = null
  private stillFrames = 0
  private wasMoving = false

  constructor(
    private calibration: Calibration,
    private config: VisionConfig,
  ) {}

  /** Adopt the current view as the empty board. Call after retrieving darts. */
  reset(frame?: GrayImage): void {
    this.reference = frame ?? null
    this.stillFrames = 0
    this.wasMoving = false
  }

  setCalibration(calibration: Calibration): void {
    this.calibration = calibration
  }

  get referenceFrame(): GrayImage | null {
    return this.reference
  }

  /** Feed one frame. */
  push(frame: GrayImage): WatcherUpdate {
    if (!this.reference) {
      this.reference = frame
      return { phase: 'waiting', motion: 0 }
    }

    const diff = differenceMask(this.reference, frame, this.config.pixelThreshold)
    const motion = changedFraction(diff)

    if (motion >= this.config.motionFraction) {
      this.wasMoving = true
      this.stillFrames = 0
      return { phase: 'moving', motion }
    }

    // Still, but nothing had moved since the last reading — nothing to do.
    if (!this.wasMoving) {
      this.stillFrames = 0
      return { phase: 'waiting', motion }
    }

    this.stillFrames += 1
    if (this.stillFrames < this.config.stableFrames) {
      return { phase: 'settling', motion }
    }

    const result = detectDart(this.reference, frame, this.calibration, this.config)
    this.wasMoving = false
    this.stillFrames = 0

    if (!result.ok) {
      // Nothing recognisable changed — most likely the player reached in and
      // took their hand away again. Re-baseline so the next throw is measured
      // against what is actually on the board now.
      this.reference = frame
      return { phase: 'waiting', failure: result.reason, motion }
    }

    // The dart is now part of the board, so it becomes the new baseline and
    // the next throw is measured against it.
    this.reference = frame
    return { phase: 'waiting', reading: result.reading, motion }
  }
}
