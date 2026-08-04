/**
 * Connected components, and working out which end of a dart-shaped blob is
 * the tip.
 */

import type { Mask } from './frameDiff'
import type { Point } from './homography'

export interface Blob {
  /** Pixel coordinates belonging to this component. */
  points: Point[]
  area: number
  centroid: Point
  bounds: { minX: number; minY: number; maxX: number; maxY: number }
}

/**
 * Label connected components using 8-connectivity.
 *
 * Iterative flood fill rather than recursion: a dart lying along the frame can
 * be thousands of pixels, which is enough to blow the call stack on a phone.
 */
export function findBlobs(mask: Mask, minArea = 1): Blob[] {
  const { width, height, data } = mask
  const seen = new Uint8Array(data.length)
  const blobs: Blob[] = []
  const stack: number[] = []

  for (let start = 0; start < data.length; start++) {
    if (data[start] !== 1 || seen[start] === 1) continue

    seen[start] = 1
    stack.length = 0
    stack.push(start)
    const points: Point[] = []
    let sumX = 0
    let sumY = 0
    let minX = width
    let minY = height
    let maxX = -1
    let maxY = -1

    while (stack.length > 0) {
      const index = stack.pop()!
      const x = index % width
      const y = (index - x) / width

      points.push({ x, y })
      sumX += x
      sumY += y
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y

      for (let dy = -1; dy <= 1; dy++) {
        const ny = y + dy
        if (ny < 0 || ny >= height) continue
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx
          if (nx < 0 || nx >= width) continue
          const neighbour = ny * width + nx
          if (data[neighbour] === 1 && seen[neighbour] === 0) {
            seen[neighbour] = 1
            stack.push(neighbour)
          }
        }
      }
    }

    if (points.length >= minArea) {
      blobs.push({
        points,
        area: points.length,
        centroid: { x: sumX / points.length, y: sumY / points.length },
        bounds: { minX, minY, maxX, maxY },
      })
    }
  }

  return blobs.sort((a, b) => b.area - a.area)
}

export interface BlobShape {
  /** Unit vector along the blob's long axis. */
  axis: Point
  /** Spread along the long axis. */
  majorSpread: number
  /** Spread across it. */
  minorSpread: number
  /**
   * Ratio of minor to major spread, in `[0, 1]`. A dart is a long thin thing,
   * so this is small; a hand or a lighting change is round, so it approaches
   * 1 and gets rejected.
   */
  compactness: number
}

/**
 * The blob's principal axis, from the eigenvector of its covariance matrix.
 *
 * For a 2×2 symmetric matrix this is closed-form, so there is no need for a
 * general eigen-solver.
 */
export function blobShape(blob: Blob): BlobShape {
  const { centroid, points } = blob
  let sxx = 0
  let syy = 0
  let sxy = 0
  for (const point of points) {
    const dx = point.x - centroid.x
    const dy = point.y - centroid.y
    sxx += dx * dx
    syy += dy * dy
    sxy += dx * dy
  }
  const n = points.length || 1
  sxx /= n
  syy /= n
  sxy /= n

  const trace = sxx + syy
  const determinant = sxx * syy - sxy * sxy
  const gap = Math.sqrt(Math.max(0, (trace * trace) / 4 - determinant))
  const major = trace / 2 + gap
  const minor = Math.max(0, trace / 2 - gap)

  // Eigenvector for the larger eigenvalue. When sxy vanishes the covariance is
  // already axis-aligned and the larger variance picks the axis directly.
  let axis: Point
  if (Math.abs(sxy) > 1e-9) {
    const vx = major - syy
    const vy = sxy
    const length = Math.hypot(vx, vy) || 1
    axis = { x: vx / length, y: vy / length }
  } else {
    axis = sxx >= syy ? { x: 1, y: 0 } : { x: 0, y: 1 }
  }

  const majorSpread = Math.sqrt(major)
  const minorSpread = Math.sqrt(minor)

  return {
    axis,
    majorSpread,
    minorSpread,
    compactness: majorSpread > 0 ? minorSpread / majorSpread : 1,
  }
}

/**
 * The two extreme points of a blob along its long axis.
 *
 * One of these is the dart's tip and the other is the flight; which is which
 * is decided by the caller, since it depends on where the board's centre is.
 */
export function axisExtremes(blob: Blob, shape: BlobShape): [Point, Point] {
  let minProjection = Infinity
  let maxProjection = -Infinity
  let low = blob.centroid
  let high = blob.centroid

  for (const point of blob.points) {
    const projection =
      (point.x - blob.centroid.x) * shape.axis.x + (point.y - blob.centroid.y) * shape.axis.y
    if (projection < minProjection) {
      minProjection = projection
      low = point
    }
    if (projection > maxProjection) {
      maxProjection = projection
      high = point
    }
  }

  return [low, high]
}
