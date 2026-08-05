/**
 * Dartboard geometry — the single source of truth.
 *
 * The predecessor to this project kept board geometry in two places that
 * disagreed (a rotation-aware path used only for drawing, and a hardcoded
 * offset used for actual scoring), so scores were only correct when the board
 * happened to be photographed at one specific orientation. Everything that
 * needs to turn a position into a score goes through this module.
 *
 * ## Coordinate conventions
 *
 * **Board space** is millimetres from the centre of the bull, with `+x` to the
 * right (toward the 6) and `+y` *downward* (toward the 3). Y-down matches
 * canvas/`ImageData` convention, so the homography in `src/vision` can map
 * image pixels straight into board space with no axis flip.
 *
 * **Bearing** is degrees clockwise from straight up (the centre of the 20),
 * in `[0, 360)`. That is how a person describes a dartboard, and it makes the
 * segment lookup a single floor division.
 */

/** Radii in mm from the centre of the bull. Standard WDF/BDO dimensions. */
export const RADIUS = {
  /** Inner bull ("double bull"), worth 50. */
  innerBull: 6.35,
  /** Outer bull ("25"). */
  outerBull: 15.9,
  /** Inner edge of the treble ring. */
  trebleInner: 99.0,
  /** Outer edge of the treble ring. */
  trebleOuter: 107.0,
  /** Inner edge of the double ring. */
  doubleInner: 162.0,
  /** Outer edge of the double ring — the edge of the scoring area. */
  doubleOuter: 170.0,
} as const

/** Number of radial segments. */
export const SEGMENT_COUNT = 20

/** Angular width of one segment, in degrees. */
export const SEGMENT_ANGLE = 360 / SEGMENT_COUNT // 18

/**
 * Segment values in clockwise order starting from the 20 at the top.
 * Index 0 is centred on bearing 0, index 5 (the 6) on bearing 90, and so on.
 */
export const SEGMENTS: readonly number[] = [
  20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5,
] as const

/** Which ring a dart landed in. */
export type Region = 'inner_bull' | 'outer_bull' | 'single' | 'treble' | 'double' | 'miss'

export interface BoardScore {
  /** The segment number hit (0 for bull, outer bull and misses). */
  base: number
  /** 1, 2 or 3. Bulls and misses are multiplier 1. */
  multiplier: number
  /** Points actually scored. */
  total: number
  region: Region
  /**
   * Index into {@link SEGMENTS}, or `null` for bulls and misses where no
   * segment applies.
   */
  segmentIndex: number | null
}

const MISS: BoardScore = Object.freeze({
  base: 0,
  multiplier: 1,
  total: 0,
  region: 'miss',
  segmentIndex: null,
})

const INNER_BULL: BoardScore = Object.freeze({
  base: 50,
  multiplier: 1,
  total: 50,
  region: 'inner_bull',
  segmentIndex: null,
})

const OUTER_BULL: BoardScore = Object.freeze({
  base: 25,
  multiplier: 1,
  total: 25,
  region: 'outer_bull',
  segmentIndex: null,
})

/** A miss — a dart outside the scoring area, or a bounce-out. */
export function miss(): BoardScore {
  return MISS
}

/**
 * Normalise any angle into `[0, 360)`.
 */
export function normaliseBearing(deg: number): number {
  const wrapped = deg % 360
  return wrapped < 0 ? wrapped + 360 : wrapped
}

/**
 * Convert board-space cartesian coordinates (mm, y-down) to a radius and a
 * bearing clockwise from the 20.
 */
export function cartesianToPolar(xMm: number, yMm: number): { radiusMm: number; bearingDeg: number } {
  const radiusMm = Math.hypot(xMm, yMm)
  // atan2(x, -y) measures clockwise from straight up in a y-down space.
  const bearingDeg = normaliseBearing((Math.atan2(xMm, -yMm) * 180) / Math.PI)
  return { radiusMm, bearingDeg }
}

/**
 * Convert a radius and bearing back to board-space cartesian coordinates.
 * Used for drawing the board and for placing corrected throws.
 */
export function polarToCartesian(radiusMm: number, bearingDeg: number): { xMm: number; yMm: number } {
  const rad = (bearingDeg * Math.PI) / 180
  return { xMm: radiusMm * Math.sin(rad), yMm: -radiusMm * Math.cos(rad) }
}

/**
 * The index into {@link SEGMENTS} for a given bearing.
 *
 * Segment 0 (the 20) is *centred* on bearing 0, so it spans 351°–360° and
 * 0°–9°. Shifting by half a segment before dividing handles that wrap.
 */
export function segmentIndexForBearing(bearingDeg: number): number {
  const shifted = normaliseBearing(bearingDeg + SEGMENT_ANGLE / 2)
  return Math.floor(shifted / SEGMENT_ANGLE) % SEGMENT_COUNT
}

/** The bearing of the centre line of a segment, for rendering. */
export function bearingForSegmentIndex(index: number): number {
  return normaliseBearing(index * SEGMENT_ANGLE)
}

/**
 * Score a dart from its position on the board.
 *
 * @param radiusMm  Distance from the centre of the bull, in mm.
 * @param bearingDeg Degrees clockwise from the 20. Any value is accepted and
 *                   normalised.
 */
export function scoreFromPolar(radiusMm: number, bearingDeg: number): BoardScore {
  if (!Number.isFinite(radiusMm) || !Number.isFinite(bearingDeg)) return MISS
  if (radiusMm < 0) return MISS

  if (radiusMm <= RADIUS.innerBull) return INNER_BULL
  if (radiusMm <= RADIUS.outerBull) return OUTER_BULL
  if (radiusMm > RADIUS.doubleOuter) return MISS

  const segmentIndex = segmentIndexForBearing(bearingDeg)
  const base = SEGMENTS[segmentIndex]!

  let multiplier: number
  let region: Region
  if (radiusMm >= RADIUS.trebleInner && radiusMm <= RADIUS.trebleOuter) {
    multiplier = 3
    region = 'treble'
  } else if (radiusMm >= RADIUS.doubleInner) {
    multiplier = 2
    region = 'double'
  } else {
    multiplier = 1
    region = 'single'
  }

  return { base, multiplier, total: base * multiplier, region, segmentIndex }
}

/** Score a dart from board-space cartesian coordinates. */
export function scoreFromCartesian(xMm: number, yMm: number): BoardScore {
  const { radiusMm, bearingDeg } = cartesianToPolar(xMm, yMm)
  return scoreFromPolar(radiusMm, bearingDeg)
}

/**
 * Build a score directly from a segment and multiplier, for manual entry and
 * for correcting a misread throw.
 */
export function scoreFromSegment(base: number, multiplier: 1 | 2 | 3): BoardScore {
  if (base === 50) return INNER_BULL
  if (base === 25) return OUTER_BULL
  if (base === 0) return MISS

  const segmentIndex = SEGMENTS.indexOf(base)
  if (segmentIndex === -1) {
    throw new Error(`${base} is not a dartboard segment`)
  }
  const region: Region = multiplier === 3 ? 'treble' : multiplier === 2 ? 'double' : 'single'
  return { base, multiplier, total: base * multiplier, region, segmentIndex }
}

/** Short display label, e.g. `T20`, `D16`, `25`, `BULL`, `MISS`. */
export function describeScore(score: BoardScore): string {
  switch (score.region) {
    case 'inner_bull':
      return 'BULL'
    case 'outer_bull':
      return '25'
    case 'miss':
      return 'MISS'
    case 'treble':
      return `T${score.base}`
    case 'double':
      return `D${score.base}`
    case 'single':
      return `${score.base}`
  }
}

/**
 * Every distinct throw a dart can make, used by the checkout solver and by
 * tests. 20 segments × 3 multipliers, plus both bulls and a miss.
 */
export function allPossibleThrows(): BoardScore[] {
  const throws: BoardScore[] = [MISS, OUTER_BULL, INNER_BULL]
  for (const base of SEGMENTS) {
    for (const multiplier of [1, 2, 3] as const) {
      throws.push(scoreFromSegment(base, multiplier))
    }
  }
  return throws
}

/** Whether a score counts as a double for double-in/double-out rules. */
export function isDouble(score: BoardScore): boolean {
  // The inner bull is the 50, which counts as a double 25 for checkout purposes.
  return score.region === 'double' || score.region === 'inner_bull'
}
