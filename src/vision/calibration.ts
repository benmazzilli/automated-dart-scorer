/**
 * Turning four taps on the camera preview into a usable board transform.
 */

import { RADIUS, polarToCartesian, scoreFromCartesian, type BoardScore } from '../game/board'
import {
  applyHomography,
  homographyFromPoints,
  invertHomography,
  type Homography,
  type Point,
} from './homography'

/**
 * The four points the player taps, in order.
 *
 * Chosen to be unambiguous on a real board and as far apart as possible: the
 * outer edge of the double ring at the four compass positions. Spreading them
 * to the rim keeps the solved transform stable — points clustered near the
 * centre would make a small tapping error swing the whole mapping.
 */
export const CALIBRATION_TARGETS = [
  { label: 'Top of the 20', bearing: 0 },
  { label: 'Right of the 6', bearing: 90 },
  { label: 'Bottom of the 3', bearing: 180 },
  { label: 'Left of the 11', bearing: 270 },
] as const

export type CalibrationPoints = readonly [Point, Point, Point, Point]

/** The board-space position of each calibration target, in millimetres. */
export const CALIBRATION_BOARD_POINTS: CalibrationPoints = CALIBRATION_TARGETS.map((target) => {
  const { xMm, yMm } = polarToCartesian(RADIUS.doubleOuter, target.bearing)
  return { x: xMm, y: yMm }
}) as unknown as CalibrationPoints

export interface Calibration {
  /** Image pixels to board millimetres. */
  imageToBoard: Homography
  /** Board millimetres to image pixels, for drawing overlays. */
  boardToImage: Homography
  /** The taps this was built from, so it can be shown and adjusted. */
  taps: CalibrationPoints
}

/**
 * Build a calibration from four taps, given in the order of
 * {@link CALIBRATION_TARGETS}.
 *
 * @returns `null` if the taps are degenerate — three in a line, or two on the
 *          same spot — which no transform can be recovered from.
 */
export function calibrate(taps: CalibrationPoints): Calibration | null {
  const imageToBoard = homographyFromPoints(taps, CALIBRATION_BOARD_POINTS)
  if (!imageToBoard) return null

  const boardToImage = invertHomography(imageToBoard)
  if (!boardToImage) return null

  return { imageToBoard, boardToImage, taps }
}

/** Map a point in the camera image to board millimetres. */
export function imagePointToBoard(calibration: Calibration, point: Point): Point {
  return applyHomography(calibration.imageToBoard, point)
}

/** Map board millimetres back to a point in the camera image. */
export function boardPointToImage(calibration: Calibration, point: Point): Point {
  return applyHomography(calibration.boardToImage, point)
}

/** Score a point in the camera image directly. */
export function scoreImagePoint(calibration: Calibration, point: Point): BoardScore {
  const board = imagePointToBoard(calibration, point)
  return scoreFromCartesian(board.x, board.y)
}

/**
 * A rough sanity check on a calibration.
 *
 * Taps that are wildly out — tapped in the wrong order, say, or on the wrong
 * features — still produce a valid transform, just a nonsensical one. Mapping
 * the centre of the tapped quadrilateral back should land near the bull, and
 * the board should not come out absurdly small or large in the image.
 */
export function calibrationLooksSane(calibration: Calibration): boolean {
  const centre = {
    x: calibration.taps.reduce((total, p) => total + p.x, 0) / 4,
    y: calibration.taps.reduce((total, p) => total + p.y, 0) / 4,
  }
  const board = imagePointToBoard(calibration, centre)
  if (!Number.isFinite(board.x) || !Number.isFinite(board.y)) return false

  // The centroid of four rim points sits at the bull for a square-on view, and
  // drifts under perspective. A third of the board radius is generous enough
  // for a steep angle but catches taps given in the wrong order.
  const driftMm = Math.hypot(board.x, board.y)
  if (driftMm > RADIUS.doubleOuter / 3) return false

  // The tapped quadrilateral has to be a reasonable size on screen, or the
  // pixel-per-millimetre scale is too coarse to place a dart tip.
  const spanPx = Math.max(
    ...calibration.taps.map((a) =>
      Math.max(...calibration.taps.map((b) => Math.hypot(a.x - b.x, a.y - b.y))),
    ),
  )
  return spanPx > 80
}
