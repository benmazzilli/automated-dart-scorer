import { describe, it, expect } from 'vitest'
import {
  RADIUS,
  SEGMENTS,
  SEGMENT_ANGLE,
  SEGMENT_COUNT,
  allPossibleThrows,
  bearingForSegmentIndex,
  cartesianToPolar,
  describeScore,
  isDouble,
  normaliseBearing,
  polarToCartesian,
  scoreFromCartesian,
  scoreFromPolar,
  scoreFromSegment,
  segmentIndexForBearing,
} from './board'

/** A radius comfortably inside the large single area between treble and double. */
const OUTER_SINGLE = 130
/** A radius inside the small single area between bull and treble. */
const INNER_SINGLE = 60

describe('segment layout', () => {
  it('has 20 unique segments summing to 210', () => {
    expect(SEGMENTS).toHaveLength(SEGMENT_COUNT)
    expect(new Set(SEGMENTS).size).toBe(SEGMENT_COUNT)
    expect(SEGMENTS.reduce((a, b) => a + b, 0)).toBe(210)
    expect([...SEGMENTS].sort((a, b) => a - b)).toEqual(
      Array.from({ length: 20 }, (_, i) => i + 1),
    )
  })

  it('places the cardinal segments where a real board has them', () => {
    // These four are the anchor points used for camera calibration, so if any
    // of them is wrong the whole vision pipeline scores wrong.
    expect(scoreFromPolar(OUTER_SINGLE, 0).base).toBe(20) // 12 o'clock
    expect(scoreFromPolar(OUTER_SINGLE, 90).base).toBe(6) // 3 o'clock
    expect(scoreFromPolar(OUTER_SINGLE, 180).base).toBe(3) // 6 o'clock
    expect(scoreFromPolar(OUTER_SINGLE, 270).base).toBe(11) // 9 o'clock
  })

  it('scores each segment correctly at its centre bearing', () => {
    SEGMENTS.forEach((value, index) => {
      const bearing = bearingForSegmentIndex(index)
      expect(scoreFromPolar(OUTER_SINGLE, bearing).base).toBe(value)
    })
  })

  it('keeps neighbouring segments adjacent all the way round', () => {
    for (let i = 0; i < SEGMENT_COUNT; i++) {
      const justInside = scoreFromPolar(
        OUTER_SINGLE,
        bearingForSegmentIndex(i) + SEGMENT_ANGLE / 2 - 0.01,
      )
      const justOutside = scoreFromPolar(
        OUTER_SINGLE,
        bearingForSegmentIndex(i) + SEGMENT_ANGLE / 2 + 0.01,
      )
      expect(justInside.base).toBe(SEGMENTS[i])
      expect(justOutside.base).toBe(SEGMENTS[(i + 1) % SEGMENT_COUNT])
    }
  })

  it('wraps the 20 across 360 degrees', () => {
    // The 20 is centred on 0, so it straddles the wrap point.
    expect(scoreFromPolar(OUTER_SINGLE, 355).base).toBe(20)
    expect(scoreFromPolar(OUTER_SINGLE, 5).base).toBe(20)
    expect(scoreFromPolar(OUTER_SINGLE, 350).base).toBe(5) // last segment
    expect(scoreFromPolar(OUTER_SINGLE, 10).base).toBe(1) // first clockwise
  })

  it('accepts unnormalised bearings', () => {
    expect(scoreFromPolar(OUTER_SINGLE, -90).base).toBe(11)
    expect(scoreFromPolar(OUTER_SINGLE, 450).base).toBe(6)
    expect(scoreFromPolar(OUTER_SINGLE, -270).base).toBe(6)
  })
})

describe('rings', () => {
  it('scores the bulls', () => {
    expect(scoreFromPolar(0, 0)).toMatchObject({ total: 50, region: 'inner_bull' })
    expect(scoreFromPolar(6, 123)).toMatchObject({ total: 50, region: 'inner_bull' })
    expect(scoreFromPolar(10, 200)).toMatchObject({ total: 25, region: 'outer_bull' })
    expect(scoreFromPolar(15, 45)).toMatchObject({ total: 25, region: 'outer_bull' })
  })

  it('scores singles, trebles and doubles on the 20', () => {
    expect(scoreFromPolar(INNER_SINGLE, 0)).toMatchObject({ total: 20, region: 'single' })
    expect(scoreFromPolar(103, 0)).toMatchObject({ total: 60, region: 'treble' })
    expect(scoreFromPolar(OUTER_SINGLE, 0)).toMatchObject({ total: 20, region: 'single' })
    expect(scoreFromPolar(166, 0)).toMatchObject({ total: 40, region: 'double' })
  })

  it('treats anything beyond the double ring as a miss', () => {
    expect(scoreFromPolar(RADIUS.doubleOuter + 0.01, 0)).toMatchObject({ total: 0, region: 'miss' })
    expect(scoreFromPolar(500, 90)).toMatchObject({ total: 0, region: 'miss' })
  })

  it('places ring boundaries on the scoring side of the wire', () => {
    // Exactly on a boundary radius should score the higher-value ring, so a
    // dart resting on the wire is never silently downgraded.
    expect(scoreFromPolar(RADIUS.innerBull, 0).region).toBe('inner_bull')
    expect(scoreFromPolar(RADIUS.outerBull, 0).region).toBe('outer_bull')
    expect(scoreFromPolar(RADIUS.trebleInner, 0).region).toBe('treble')
    expect(scoreFromPolar(RADIUS.trebleOuter, 0).region).toBe('treble')
    expect(scoreFromPolar(RADIUS.doubleInner, 0).region).toBe('double')
    expect(scoreFromPolar(RADIUS.doubleOuter, 0).region).toBe('double')
  })

  it('reverts to single just outside each scoring ring', () => {
    expect(scoreFromPolar(RADIUS.outerBull + 0.01, 0).region).toBe('single')
    expect(scoreFromPolar(RADIUS.trebleInner - 0.01, 0).region).toBe('single')
    expect(scoreFromPolar(RADIUS.trebleOuter + 0.01, 0).region).toBe('single')
    expect(scoreFromPolar(RADIUS.doubleInner - 0.01, 0).region).toBe('single')
  })

  it('rejects nonsense input as a miss rather than throwing', () => {
    expect(scoreFromPolar(NaN, 0).region).toBe('miss')
    expect(scoreFromPolar(50, NaN).region).toBe('miss')
    expect(scoreFromPolar(-1, 0).region).toBe('miss')
    expect(scoreFromPolar(Infinity, 0).region).toBe('miss')
  })

  it('produces the maximum three-dart score of 180', () => {
    const t20 = scoreFromPolar(103, 0)
    expect(t20.total * 3).toBe(180)
  })
})

describe('coordinate conversion', () => {
  it('maps cartesian directions to the right segments', () => {
    // +y is downward, so the 20 is at negative y.
    expect(scoreFromCartesian(0, -OUTER_SINGLE).base).toBe(20)
    expect(scoreFromCartesian(OUTER_SINGLE, 0).base).toBe(6)
    expect(scoreFromCartesian(0, OUTER_SINGLE).base).toBe(3)
    expect(scoreFromCartesian(-OUTER_SINGLE, 0).base).toBe(11)
  })

  it('round-trips polar to cartesian and back', () => {
    for (let bearing = 0; bearing < 360; bearing += 7) {
      for (const radius of [5, 30, 103, 130, 166]) {
        const { xMm, yMm } = polarToCartesian(radius, bearing)
        const back = cartesianToPolar(xMm, yMm)
        expect(back.radiusMm).toBeCloseTo(radius, 6)
        expect(back.bearingDeg).toBeCloseTo(bearing, 6)
      }
    }
  })

  it('agrees between the polar and cartesian scoring entry points', () => {
    // Offset by half a degree to stay off exact segment boundaries. Landing
    // precisely on a wire is ambiguous by definition — the round trip through
    // sin/cos/atan2 can put the result either side of the boundary by one ulp,
    // and no amount of care in the code can decide which side a dart resting on
    // the wire "really" hit. Boundary behaviour is pinned separately below.
    for (let bearing = 0.5; bearing < 360; bearing += 3) {
      for (const radius of [0, 10, 60, 103, 130, 166, 200]) {
        const { xMm, yMm } = polarToCartesian(radius, bearing)
        expect(scoreFromCartesian(xMm, yMm)).toEqual(scoreFromPolar(radius, bearing))
      }
    }
  })

  it('is stable either side of a segment boundary', () => {
    // The boundary between the 20 and the 1 sits at bearing 9. A dart a
    // hundredth of a degree either side must land in a definite, correct
    // segment — only the exact boundary itself is undefined.
    for (const epsilon of [0.01, 0.5, 1]) {
      expect(scoreFromPolar(OUTER_SINGLE, 9 - epsilon).base).toBe(20)
      expect(scoreFromPolar(OUTER_SINGLE, 9 + epsilon).base).toBe(1)
    }
    // And the same through a cartesian round trip.
    const justInside = polarToCartesian(OUTER_SINGLE, 8.9)
    const justOutside = polarToCartesian(OUTER_SINGLE, 9.1)
    expect(scoreFromCartesian(justInside.xMm, justInside.yMm).base).toBe(20)
    expect(scoreFromCartesian(justOutside.xMm, justOutside.yMm).base).toBe(1)
  })

  it('normalises bearings into [0, 360)', () => {
    expect(normaliseBearing(0)).toBe(0)
    expect(normaliseBearing(360)).toBe(0)
    expect(normaliseBearing(-1)).toBe(359)
    expect(normaliseBearing(721)).toBe(1)
  })

  it('puts the centre of the bull at radius zero', () => {
    expect(cartesianToPolar(0, 0).radiusMm).toBe(0)
  })
})

describe('scoreFromSegment', () => {
  it('builds scores matching the geometric path', () => {
    expect(scoreFromSegment(20, 3)).toMatchObject({ total: 60, region: 'treble' })
    expect(scoreFromSegment(16, 2)).toMatchObject({ total: 32, region: 'double' })
    expect(scoreFromSegment(5, 1)).toMatchObject({ total: 5, region: 'single' })
    expect(scoreFromSegment(50, 1)).toMatchObject({ total: 50, region: 'inner_bull' })
    expect(scoreFromSegment(25, 1)).toMatchObject({ total: 25, region: 'outer_bull' })
    expect(scoreFromSegment(0, 1)).toMatchObject({ total: 0, region: 'miss' })
  })

  it('agrees with the geometric path for every segment and multiplier', () => {
    SEGMENTS.forEach((value, index) => {
      const bearing = bearingForSegmentIndex(index)
      expect(scoreFromSegment(value, 1)).toEqual(scoreFromPolar(OUTER_SINGLE, bearing))
      expect(scoreFromSegment(value, 3)).toEqual(scoreFromPolar(103, bearing))
      expect(scoreFromSegment(value, 2)).toEqual(scoreFromPolar(166, bearing))
    })
  })

  it('rejects numbers that are not on the board', () => {
    expect(() => scoreFromSegment(21, 1)).toThrow(/not a dartboard segment/)
    expect(() => scoreFromSegment(-3, 1)).toThrow()
  })
})

describe('helpers', () => {
  it('labels scores for display', () => {
    expect(describeScore(scoreFromSegment(20, 3))).toBe('T20')
    expect(describeScore(scoreFromSegment(16, 2))).toBe('D16')
    expect(describeScore(scoreFromSegment(7, 1))).toBe('7')
    expect(describeScore(scoreFromSegment(50, 1))).toBe('BULL')
    expect(describeScore(scoreFromSegment(25, 1))).toBe('25')
    expect(describeScore(scoreFromSegment(0, 1))).toBe('MISS')
  })

  it('counts doubles and the bull as doubles, but not trebles or 25', () => {
    expect(isDouble(scoreFromSegment(16, 2))).toBe(true)
    expect(isDouble(scoreFromSegment(50, 1))).toBe(true) // 50 is the double 25
    expect(isDouble(scoreFromSegment(25, 1))).toBe(false)
    expect(isDouble(scoreFromSegment(20, 3))).toBe(false)
    expect(isDouble(scoreFromSegment(20, 1))).toBe(false)
  })

  it('enumerates 63 distinct throws', () => {
    const throws = allPossibleThrows()
    expect(throws).toHaveLength(20 * 3 + 3) // segments, both bulls, and a miss
    expect(new Set(throws.map(describeScore)).size).toBe(throws.length)
    expect(Math.max(...throws.map((t) => t.total))).toBe(60)
  })

  it('maps bearings to segment indices consistently', () => {
    for (let i = 0; i < SEGMENT_COUNT; i++) {
      expect(segmentIndexForBearing(bearingForSegmentIndex(i))).toBe(i)
    }
  })
})
