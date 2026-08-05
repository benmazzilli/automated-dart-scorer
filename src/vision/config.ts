/**
 * Every tunable threshold in the vision pipeline, in one place.
 *
 * These were set against synthetic frames and will need adjusting against a
 * real board, real lighting and a real phone. That is expected, not a defect:
 * the debug overlay shows the live difference mask, the detected blob and the
 * computed tip so tuning is a visible feedback loop rather than guesswork.
 */
export interface VisionConfig {
  /** Working resolution frames are sampled down to. Smaller is faster. */
  frameWidth: number
  frameHeight: number

  /** Per-pixel brightness change counted as "different", 0–255. */
  pixelThreshold: number

  /**
   * Fraction of pixels that must change for the scene to count as moving.
   * Above this, something is happening — a hand, a dart in flight, a player
   * walking past — and no reading is taken.
   */
  motionFraction: number

  /** Consecutive still frames before the scene is treated as settled. */
  stableFrames: number

  /** Smallest blob, in pixels, that can be a dart. Rejects sensor noise. */
  minBlobArea: number
  /** Largest blob that can be a dart. Rejects a hand or an arm. */
  maxBlobArea: number

  /**
   * Fraction of the difference that must be inside the board for a settled
   * frame to be read. Stops a change entirely off the board being scored.
   */
  minOnBoardFraction: number

  /** Reject a blob rounder than this; a dart is elongated. */
  maxCompactness: number

  /** Frames per second pulled from the camera. */
  captureFps: number

  /** Below this confidence the reading is offered for confirmation, not auto-accepted. */
  autoAcceptConfidence: number
}

export const DEFAULT_VISION_CONFIG: VisionConfig = {
  frameWidth: 640,
  frameHeight: 480,
  pixelThreshold: 34,
  motionFraction: 0.004,
  stableFrames: 5,
  minBlobArea: 40,
  maxBlobArea: 6000,
  minOnBoardFraction: 0.5,
  maxCompactness: 0.82,
  captureFps: 10,
  autoAcceptConfidence: 0.6,
}
