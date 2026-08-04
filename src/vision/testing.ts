/**
 * Synthetic frames for testing the vision pipeline.
 *
 * With no video of a real board to work from, the pipeline is developed
 * against generated frames: a textured backdrop standing in for the board, and
 * darts drawn at known board positions. That verifies the geometry, the
 * detection and the tip-picking logic are correct; it does not verify the
 * thresholds survive real lighting, which only a real board can settle.
 */

import { RADIUS, polarToCartesian } from '../game/board'
import { boardPointToImage, calibrate, type Calibration } from './calibration'
import type { GrayImage } from './frameDiff'
import type { Point } from './homography'

export interface SyntheticScene {
  width: number
  height: number
  calibration: Calibration
}

/**
 * A camera looking square-on at a board, with a little perspective so the
 * transform being solved is not the trivial one.
 */
export function makeScene(width = 640, height = 480, skew = 0): SyntheticScene {
  const cx = width / 2
  const cy = height / 2
  const r = Math.min(width, height) * 0.42

  // Taps in the order the calibration expects: top of 20, right of 6, bottom
  // of 3, left of 11. `skew` squashes the top edge to fake a raised camera.
  const taps = [
    { x: cx, y: cy - r * (1 - skew) },
    { x: cx + r, y: cy },
    { x: cx, y: cy + r },
    { x: cx - r, y: cy },
  ] as const

  const calibration = calibrate(taps)
  if (!calibration) throw new Error('synthetic scene produced a degenerate calibration')
  return { width, height, calibration }
}

/** A repeatable pseudo-random source, so failures are reproducible. */
function makeRandom(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0
    return state / 0xffffffff
  }
}

/**
 * A backdrop with enough texture to be realistic, but stable frame to frame.
 * Noise is added separately so a "still" scene can be made genuinely still or
 * slightly noisy on demand.
 */
export function makeBackground(scene: SyntheticScene, seed = 1): GrayImage {
  const { width, height } = scene
  const data = new Uint8Array(width * height)
  const random = makeRandom(seed)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const wedge = Math.sin(x * 0.07) * 12 + Math.cos(y * 0.09) * 12
      data[y * width + x] = Math.max(0, Math.min(255, 96 + wedge + random() * 6))
    }
  }
  return { data, width, height }
}

/** Copy an image so a frame can be drawn on without disturbing the original. */
export function cloneImage(image: GrayImage): GrayImage {
  return { data: new Uint8Array(image.data), width: image.width, height: image.height }
}

/** Add uniform noise, to check thresholds are not razor-thin. */
export function addNoise(image: GrayImage, amplitude: number, seed = 7): GrayImage {
  const out = cloneImage(image)
  const random = makeRandom(seed)
  for (let i = 0; i < out.data.length; i++) {
    const value = out.data[i]! + (random() - 0.5) * 2 * amplitude
    out.data[i] = Math.max(0, Math.min(255, value))
  }
  return out
}

function drawDisc(image: GrayImage, centre: Point, radius: number, value: number): void {
  const minX = Math.max(0, Math.floor(centre.x - radius))
  const maxX = Math.min(image.width - 1, Math.ceil(centre.x + radius))
  const minY = Math.max(0, Math.floor(centre.y - radius))
  const maxY = Math.min(image.height - 1, Math.ceil(centre.y + radius))
  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      if (Math.hypot(x - centre.x, y - centre.y) <= radius) {
        image.data[y * image.width + x] = value
      }
    }
  }
}

export interface DrawDartOptions {
  /** Board position of the tip, in millimetres. */
  radiusMm: number
  bearingDeg: number
  /** Length of the visible dart in pixels, tip to flight. */
  lengthPx?: number
  thicknessPx?: number
  /** Brightness of the dart against the backdrop. */
  value?: number
}

/**
 * Draw a dart onto a frame and return where its tip is in board millimetres.
 *
 * The dart is drawn pointing *outward* from the board centre, which is how a
 * dart appears to a camera in front of the board: the flight is nearer the
 * lens, so it projects further out than the tip.
 */
export function drawDart(
  scene: SyntheticScene,
  frame: GrayImage,
  options: DrawDartOptions,
): { tipBoard: Point; tipImage: Point } {
  const { radiusMm, bearingDeg, lengthPx = 34, thicknessPx = 3.2, value = 245 } = options

  const tipMm = polarToCartesian(radiusMm, bearingDeg)
  const tipBoard = { x: tipMm.xMm, y: tipMm.yMm }
  const tipImage = boardPointToImage(scene.calibration, tipBoard)

  // Direction away from the board centre, in image space.
  const outwardMm = polarToCartesian(radiusMm + 20, bearingDeg)
  const outwardImage = boardPointToImage(scene.calibration, { x: outwardMm.xMm, y: outwardMm.yMm })
  let dx = outwardImage.x - tipImage.x
  let dy = outwardImage.y - tipImage.y
  const length = Math.hypot(dx, dy) || 1
  dx /= length
  dy /= length

  const steps = Math.ceil(lengthPx * 2)
  for (let i = 0; i <= steps; i++) {
    const t = (i / steps) * lengthPx
    drawDisc(frame, { x: tipImage.x + dx * t, y: tipImage.y + dy * t }, thicknessPx, value)
  }

  return { tipBoard, tipImage }
}

/** A frame with the given darts on it. */
export function frameWithDarts(
  scene: SyntheticScene,
  background: GrayImage,
  darts: DrawDartOptions[],
): GrayImage {
  const frame = cloneImage(background)
  for (const dart of darts) drawDart(scene, frame, dart)
  return frame
}

/** A large blurry shape standing in for a hand reaching across the board. */
export function drawHand(scene: SyntheticScene, frame: GrayImage): GrayImage {
  const out = cloneImage(frame)
  const centre = boardPointToImage(scene.calibration, { x: 0, y: RADIUS.doubleOuter * 0.3 })
  drawDisc(out, centre, 55, 30)
  return out
}
