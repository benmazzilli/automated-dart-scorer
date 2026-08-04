/**
 * Perspective transforms from four point correspondences.
 *
 * This is what turns "where the dart appears in the camera image" into "where
 * the dart is on the board", and it is why calibration is four taps rather
 * than an attempt to find the board automatically. Because the taps define
 * the board's orientation directly, the rotation problem that made the
 * previous implementation score wrongly cannot arise: there is no separate
 * angular offset to get out of step.
 *
 * Written out longhand rather than pulled from OpenCV.js, which is an 8–10MB
 * WebAssembly download — unacceptable for a phone app whose whole point is
 * loading instantly at the oche.
 */

export interface Point {
  x: number
  y: number
}

/**
 * A 3×3 homography in row-major order. The bottom-right element is fixed at 1,
 * which is the usual normalisation and costs no generality.
 */
export type Homography = readonly [
  number, number, number,
  number, number, number,
  number, number, number,
]

/**
 * Solve a dense linear system by Gaussian elimination with partial pivoting.
 *
 * @param matrix `n × n`, row-major. Modified in place.
 * @param vector Right-hand side of length `n`. Modified in place.
 * @returns The solution, or `null` if the system is singular.
 */
export function solveLinearSystem(
  matrix: number[],
  vector: number[],
  n: number,
): number[] | null {
  for (let column = 0; column < n; column++) {
    // Partial pivoting: use the largest remaining value in this column as the
    // pivot. Without it, a calibration tap that makes one coefficient tiny
    // turns into a division that blows the whole solution up.
    let pivotRow = column
    let best = Math.abs(matrix[column * n + column] ?? 0)
    for (let row = column + 1; row < n; row++) {
      const candidate = Math.abs(matrix[row * n + column] ?? 0)
      if (candidate > best) {
        best = candidate
        pivotRow = row
      }
    }
    if (best < 1e-12) return null

    if (pivotRow !== column) {
      for (let k = 0; k < n; k++) {
        const a = matrix[column * n + k]!
        matrix[column * n + k] = matrix[pivotRow * n + k]!
        matrix[pivotRow * n + k] = a
      }
      const b = vector[column]!
      vector[column] = vector[pivotRow]!
      vector[pivotRow] = b
    }

    const pivot = matrix[column * n + column]!
    for (let row = column + 1; row < n; row++) {
      const factor = matrix[row * n + column]! / pivot
      if (factor === 0) continue
      for (let k = column; k < n; k++) {
        matrix[row * n + k] = matrix[row * n + k]! - factor * matrix[column * n + k]!
      }
      vector[row] = vector[row]! - factor * vector[column]!
    }
  }

  const solution = new Array<number>(n).fill(0)
  for (let row = n - 1; row >= 0; row--) {
    let sum = vector[row]!
    for (let k = row + 1; k < n; k++) sum -= matrix[row * n + k]! * solution[k]!
    solution[row] = sum / matrix[row * n + row]!
  }
  return solution
}

/**
 * Build the homography mapping four source points onto four destination
 * points.
 *
 * @returns The transform, or `null` if the points are degenerate — three in a
 *          line, or two on top of each other, which is what a careless
 *          calibration tap produces.
 */
export function homographyFromPoints(
  source: readonly [Point, Point, Point, Point],
  destination: readonly [Point, Point, Point, Point],
): Homography | null {
  // Each correspondence contributes two rows:
  //   h11·x + h12·y + h13 − h31·x·u − h32·y·u = u
  //   h21·x + h22·y + h23 − h31·x·v − h32·y·v = v
  const matrix: number[] = []
  const vector: number[] = []

  for (let i = 0; i < 4; i++) {
    const { x, y } = source[i]!
    const { x: u, y: v } = destination[i]!
    matrix.push(x, y, 1, 0, 0, 0, -x * u, -y * u)
    vector.push(u)
    matrix.push(0, 0, 0, x, y, 1, -x * v, -y * v)
    vector.push(v)
  }

  const h = solveLinearSystem(matrix, vector, 8)
  if (!h) return null
  if (h.some((value) => !Number.isFinite(value))) return null

  return [h[0]!, h[1]!, h[2]!, h[3]!, h[4]!, h[5]!, h[6]!, h[7]!, 1]
}

/** Apply a homography to a point. */
export function applyHomography(h: Homography, point: Point): Point {
  const denominator = h[6] * point.x + h[7] * point.y + h[8]
  if (Math.abs(denominator) < 1e-12) return { x: NaN, y: NaN }
  return {
    x: (h[0] * point.x + h[1] * point.y + h[2]) / denominator,
    y: (h[3] * point.x + h[4] * point.y + h[5]) / denominator,
  }
}

/**
 * Invert a homography, for mapping board coordinates back into the image —
 * used to draw the calibration guide over the camera preview.
 */
export function invertHomography(h: Homography): Homography | null {
  const [a, b, c, d, e, f, g, i, j] = h

  const A = e * j - f * i
  const B = f * g - d * j
  const C = d * i - e * g
  const determinant = a * A + b * B + c * C
  if (Math.abs(determinant) < 1e-12) return null

  const inverse = [
    A,
    c * i - b * j,
    b * f - c * e,
    B,
    a * j - c * g,
    c * d - a * f,
    C,
    b * g - a * i,
    a * e - b * d,
  ].map((value) => value / determinant)

  // Renormalise so the bottom-right element is 1 again.
  const scale = inverse[8]!
  if (Math.abs(scale) < 1e-12) return null
  const normalised = inverse.map((value) => value / scale)

  return [
    normalised[0]!, normalised[1]!, normalised[2]!,
    normalised[3]!, normalised[4]!, normalised[5]!,
    normalised[6]!, normalised[7]!, 1,
  ]
}
