import { describe, it, expect } from 'vitest'
import { RADIUS, describeScore, polarToCartesian } from '../game/board'
import { blobShape, findBlobs, axisExtremes } from './blobs'
import {
  CALIBRATION_BOARD_POINTS,
  calibrate,
  calibrationLooksSane,
  imagePointToBoard,
  scoreImagePoint,
} from './calibration'
import { DEFAULT_VISION_CONFIG } from './config'
import {
  close,
  differenceMask,
  dilate,
  erode,
  open,
  toGrayscale,
  type Mask,
} from './frameDiff'
import {
  applyHomography,
  homographyFromPoints,
  invertHomography,
  solveLinearSystem,
  type Point,
} from './homography'
import { ThrowWatcher, detectDart } from './pipeline'
import {
  addNoise,
  cloneImage,
  drawHand,
  frameWithDarts,
  makeBackground,
  makeScene,
} from './testing'

const config = DEFAULT_VISION_CONFIG

function mask(rows: string[]): Mask {
  const height = rows.length
  const width = rows[0]!.length
  const data = new Uint8Array(width * height)
  let count = 0
  rows.forEach((row, y) => {
    ;[...row].forEach((cell, x) => {
      if (cell === '#') {
        data[y * width + x] = 1
        count++
      }
    })
  })
  return { data, width, height, count }
}

const render = (m: Mask): string[] =>
  Array.from({ length: m.height }, (_, y) =>
    Array.from({ length: m.width }, (_, x) => (m.data[y * m.width + x] === 1 ? '#' : '.')).join(''),
  )

describe('solveLinearSystem', () => {
  it('solves a small system', () => {
    // 2x + y = 5, x - y = 1  =>  x = 2, y = 1
    const solution = solveLinearSystem([2, 1, 1, -1], [5, 1], 2)
    expect(solution![0]).toBeCloseTo(2, 10)
    expect(solution![1]).toBeCloseTo(1, 10)
  })

  it('returns null for a singular system', () => {
    expect(solveLinearSystem([1, 2, 2, 4], [1, 2], 2)).toBeNull()
  })

  it('pivots rather than dividing by a zero leading coefficient', () => {
    // The first pivot is 0, so this only works if rows are swapped.
    const solution = solveLinearSystem([0, 1, 1, 0], [3, 4], 2)
    expect(solution![0]).toBeCloseTo(4, 10)
    expect(solution![1]).toBeCloseTo(3, 10)
  })
})

describe('homography', () => {
  const unitSquare: readonly [Point, Point, Point, Point] = [
    { x: 0, y: 0 },
    { x: 1, y: 0 },
    { x: 1, y: 1 },
    { x: 0, y: 1 },
  ]

  it('recovers a pure scale and translation', () => {
    const target: readonly [Point, Point, Point, Point] = [
      { x: 10, y: 20 },
      { x: 30, y: 20 },
      { x: 30, y: 40 },
      { x: 10, y: 40 },
    ]
    const h = homographyFromPoints(unitSquare, target)!
    expect(h).not.toBeNull()
    for (let i = 0; i < 4; i++) {
      const mapped = applyHomography(h, unitSquare[i]!)
      expect(mapped.x).toBeCloseTo(target[i]!.x, 8)
      expect(mapped.y).toBeCloseTo(target[i]!.y, 8)
    }
    // The centre maps to the centre under an affine transform.
    const centre = applyHomography(h, { x: 0.5, y: 0.5 })
    expect(centre.x).toBeCloseTo(20, 8)
    expect(centre.y).toBeCloseTo(30, 8)
  })

  it('recovers a genuine perspective transform', () => {
    const trapezium: readonly [Point, Point, Point, Point] = [
      { x: 20, y: 0 },
      { x: 80, y: 0 },
      { x: 100, y: 60 },
      { x: 0, y: 60 },
    ]
    const h = homographyFromPoints(unitSquare, trapezium)!
    for (let i = 0; i < 4; i++) {
      const mapped = applyHomography(h, unitSquare[i]!)
      expect(mapped.x).toBeCloseTo(trapezium[i]!.x, 6)
      expect(mapped.y).toBeCloseTo(trapezium[i]!.y, 6)
    }
    // Perspective is non-affine, so the centre must NOT land at the centroid.
    const centre = applyHomography(h, { x: 0.5, y: 0.5 })
    const centroid = { x: 50, y: 30 }
    expect(Math.abs(centre.y - centroid.y)).toBeGreaterThan(0.5)
  })

  it('rejects degenerate points', () => {
    const collinear: readonly [Point, Point, Point, Point] = [
      { x: 0, y: 0 },
      { x: 1, y: 1 },
      { x: 2, y: 2 },
      { x: 3, y: 3 },
    ]
    expect(homographyFromPoints(collinear, unitSquare)).toBeNull()
    const duplicated: readonly [Point, Point, Point, Point] = [
      { x: 0, y: 0 },
      { x: 0, y: 0 },
      { x: 1, y: 1 },
      { x: 0, y: 1 },
    ]
    expect(homographyFromPoints(duplicated, unitSquare)).toBeNull()
  })

  it('inverts back to the original points', () => {
    const trapezium: readonly [Point, Point, Point, Point] = [
      { x: 20, y: 0 },
      { x: 80, y: 5 },
      { x: 100, y: 60 },
      { x: 0, y: 55 },
    ]
    const h = homographyFromPoints(unitSquare, trapezium)!
    const inverse = invertHomography(h)!
    for (const point of [{ x: 0.25, y: 0.75 }, { x: 0.5, y: 0.5 }, { x: 0.9, y: 0.1 }]) {
      const round = applyHomography(inverse, applyHomography(h, point))
      expect(round.x).toBeCloseTo(point.x, 6)
      expect(round.y).toBeCloseTo(point.y, 6)
    }
  })
})

describe('calibration', () => {
  const scene = makeScene()

  it('maps the tapped points onto the double ring', () => {
    for (let i = 0; i < 4; i++) {
      const board = imagePointToBoard(scene.calibration, scene.calibration.taps[i]!)
      expect(board.x).toBeCloseTo(CALIBRATION_BOARD_POINTS[i]!.x, 6)
      expect(board.y).toBeCloseTo(CALIBRATION_BOARD_POINTS[i]!.y, 6)
    }
  })

  it('scores the four calibration bearings as the right doubles', () => {
    // Scored a few millimetres inside the rim rather than exactly on it. The
    // taps land precisely on radius 170 by construction, which is the outer
    // wire — the same on-the-boundary ambiguity as a dart resting on a wire,
    // where a float in the last bit decides double or miss. Real calibration
    // taps are approximate and never sit exactly on it.
    const scores = [0, 90, 180, 270].map((bearing) => {
      const mm = polarToCartesian(166, bearing)
      const image = applyHomography(scene.calibration.boardToImage, { x: mm.xMm, y: mm.yMm })
      return describeScore(scoreImagePoint(scene.calibration, image))
    })
    expect(scores).toEqual(['D20', 'D6', 'D3', 'D11'])
  })

  it('puts the centre of the tapped square on the bull', () => {
    const centre = {
      x: scene.calibration.taps.reduce((t, p) => t + p.x, 0) / 4,
      y: scene.calibration.taps.reduce((t, p) => t + p.y, 0) / 4,
    }
    expect(describeScore(scoreImagePoint(scene.calibration, centre))).toBe('BULL')
  })

  it('scores every segment correctly through the transform', () => {
    // The real test of the transform: walk round the board in board space,
    // convert to image pixels, and score back through the homography.
    for (let bearing = 0; bearing < 360; bearing += 3) {
      for (const [radius, expected] of [
        [130, 'single'],
        [103, 'treble'],
        [166, 'double'],
      ] as const) {
        const mm = polarToCartesian(radius, bearing + 1.5)
        const image = applyHomography(scene.calibration.boardToImage, { x: mm.xMm, y: mm.yMm })
        const score = scoreImagePoint(scene.calibration, image)
        expect(score.region, `r=${radius} bearing=${bearing}`).toBe(expected)
      }
    }
  })

  it('works through a perspective view', () => {
    const angled = makeScene(640, 480, 0.22)
    for (let bearing = 0; bearing < 360; bearing += 9) {
      const mm = polarToCartesian(130, bearing + 1.5)
      const image = applyHomography(angled.calibration.boardToImage, { x: mm.xMm, y: mm.yMm })
      const score = scoreImagePoint(angled.calibration, image)
      const direct = polarToCartesian(130, bearing + 1.5)
      expect(score.base).toBe(
        scoreImagePoint(angled.calibration, applyHomography(angled.calibration.boardToImage, { x: direct.xMm, y: direct.yMm })).base,
      )
      expect(score.region).toBe('single')
    }
  })

  it('rejects taps that cannot be a board', () => {
    expect(calibrate([{ x: 0, y: 0 }, { x: 1, y: 1 }, { x: 2, y: 2 }, { x: 3, y: 3 }])).toBeNull()
  })

  it('flags taps given in the wrong order as not sane', () => {
    const good = makeScene().calibration
    expect(calibrationLooksSane(good)).toBe(true)

    // Swap two adjacent taps: still solvable, but a bow-tie, not a board.
    const swapped = calibrate([
      good.taps[1]!,
      good.taps[0]!,
      good.taps[2]!,
      good.taps[3]!,
    ])
    expect(swapped === null || !calibrationLooksSane(swapped)).toBe(true)
  })

  it('rejects a board tapped far too small to read', () => {
    const tiny = calibrate([
      { x: 100, y: 90 },
      { x: 110, y: 100 },
      { x: 100, y: 110 },
      { x: 90, y: 100 },
    ])
    expect(tiny).not.toBeNull()
    expect(calibrationLooksSane(tiny!)).toBe(false)
  })
})

describe('greyscale and differencing', () => {
  it('converts RGBA to luma', () => {
    const rgba = new Uint8ClampedArray([255, 255, 255, 255, 0, 0, 0, 255])
    const gray = toGrayscale(rgba, 2, 1)
    expect(gray.data[0]).toBeGreaterThan(250)
    expect(gray.data[1]).toBe(0)
  })

  it('marks only pixels past the threshold', () => {
    const a = { data: new Uint8Array([0, 100, 100]), width: 3, height: 1 }
    const b = { data: new Uint8Array([0, 110, 200]), width: 3, height: 1 }
    const diff = differenceMask(a, b, 30)
    expect([...diff.data]).toEqual([0, 0, 1])
    expect(diff.count).toBe(1)
  })

  it('refuses mismatched sizes', () => {
    const a = { data: new Uint8Array(4), width: 2, height: 2 }
    const b = { data: new Uint8Array(6), width: 3, height: 2 }
    expect(() => differenceMask(a, b, 10)).toThrow(/different sizes/)
  })
})

describe('morphology', () => {
  it('erodes away an isolated pixel', () => {
    const result = erode(mask(['.....', '..#..', '.....']))
    expect(result.count).toBe(0)
  })

  it('dilates a single pixel into a 3x3 block around it', () => {
    const result = dilate(mask(['.....', '..#..', '.....']))
    expect(render(result)).toEqual(['.###.', '.###.', '.###.'])
    expect(result.count).toBe(9)
  })

  it('opens away speckle but keeps a solid shape', () => {
    const speckled = mask([
      '.......',
      '.###.#.',
      '.###...',
      '.###...',
      '.......',
    ])
    const result = open(speckled)
    // The lone pixel is gone; the block survives.
    expect(result.data[1 * 7 + 5]).toBe(0)
    expect(result.count).toBeGreaterThan(4)
  })

  it('closes a gap in a shape', () => {
    const broken = mask(['.......', '.##.##.', '.......'])
    const result = close(broken)
    expect(result.data[1 * 7 + 3]).toBe(1)
  })
})

describe('blobs', () => {
  it('separates disconnected components, largest first', () => {
    const blobs = findBlobs(
      mask([
        '##...',
        '##...',
        '.....',
        '...#.',
      ]),
    )
    expect(blobs).toHaveLength(2)
    expect(blobs[0]!.area).toBe(4)
    expect(blobs[1]!.area).toBe(1)
  })

  it('joins diagonally touching pixels', () => {
    const blobs = findBlobs(mask(['#..', '.#.', '..#']))
    expect(blobs).toHaveLength(1)
    expect(blobs[0]!.area).toBe(3)
  })

  it('drops blobs below the minimum area', () => {
    expect(findBlobs(mask(['#..', '...', '..#']), 2)).toHaveLength(0)
  })

  it('finds the long axis of a horizontal bar', () => {
    const shape = blobShape(findBlobs(mask(['.......', '.#####.', '.......']))[0]!)
    expect(Math.abs(shape.axis.x)).toBeCloseTo(1, 5)
    expect(Math.abs(shape.axis.y)).toBeCloseTo(0, 5)
    expect(shape.compactness).toBeLessThan(0.2)
  })

  it('finds the long axis of a vertical bar', () => {
    const shape = blobShape(findBlobs(mask(['.#.', '.#.', '.#.', '.#.']))[0]!)
    expect(Math.abs(shape.axis.y)).toBeCloseTo(1, 5)
    expect(shape.compactness).toBeLessThan(0.2)
  })

  it('reports a square as compact', () => {
    const shape = blobShape(findBlobs(mask(['###', '###', '###']))[0]!)
    expect(shape.compactness).toBeCloseTo(1, 5)
  })

  it('returns the two ends of a bar', () => {
    const blob = findBlobs(mask(['.......', '.#####.', '.......']))[0]!
    const [low, high] = axisExtremes(blob, blobShape(blob))
    expect(Math.min(low.x, high.x)).toBe(1)
    expect(Math.max(low.x, high.x)).toBe(5)
  })
})

describe('detectDart', () => {
  const scene = makeScene()
  const background = makeBackground(scene)

  it('finds a treble 20 and scores it', () => {
    const withDart = frameWithDarts(scene, background, [{ radiusMm: 103, bearingDeg: 0 }])
    const result = detectDart(background, withDart, scene.calibration, config)

    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(describeScore(result.reading.score)).toBe('T20')
    expect(result.reading.confidence).toBeGreaterThan(0.5)
  })

  it('places the tip within a few millimetres of the truth', () => {
    const withDart = frameWithDarts(scene, background, [{ radiusMm: 130, bearingDeg: 45 }])
    const result = detectDart(background, withDart, scene.calibration, config)
    expect(result.ok).toBe(true)
    if (!result.ok) return

    const expected = polarToCartesian(130, 45)
    const error = Math.hypot(
      result.reading.boardPoint.x - expected.xMm,
      result.reading.boardPoint.y - expected.yMm,
    )
    expect(error).toBeLessThan(8)
  })

  it('scores darts all the way round the board', () => {
    let correct = 0
    const total = 20
    for (let i = 0; i < total; i++) {
      const bearing = i * 18 // centre of each segment
      const withDart = frameWithDarts(scene, background, [{ radiusMm: 130, bearingDeg: bearing }])
      const result = detectDart(background, withDart, scene.calibration, config)
      if (result.ok && result.reading.score.region === 'single') {
        const expected = scoreImagePoint(
          scene.calibration,
          applyHomography(scene.calibration.boardToImage, {
            x: polarToCartesian(130, bearing).xMm,
            y: polarToCartesian(130, bearing).yMm,
          }),
        )
        if (result.reading.score.base === expected.base) correct++
      }
    }
    expect(correct).toBe(total)
  })

  it('picks the tip, not the flight', () => {
    // The dart is drawn pointing outward, so choosing the wrong end would put
    // the score in the double ring or off the board entirely.
    const withDart = frameWithDarts(scene, background, [
      { radiusMm: 103, bearingDeg: 0, lengthPx: 60 },
    ])
    const result = detectDart(background, withDart, scene.calibration, config)
    expect(result.ok).toBe(true)
    if (!result.ok) return
    const radius = Math.hypot(result.reading.boardPoint.x, result.reading.boardPoint.y)
    expect(radius).toBeLessThan(RADIUS.trebleOuter + 6)
  })

  it('reports no change for an identical frame', () => {
    const result = detectDart(background, cloneImage(background), scene.calibration, config)
    expect(result.ok).toBe(false)
    if (result.ok) return
    expect(result.reason).toBe('no-change')
  })

  it('survives sensor noise', () => {
    const noisyReference = addNoise(background, 6, 11)
    const withDart = addNoise(
      frameWithDarts(scene, background, [{ radiusMm: 103, bearingDeg: 0 }]),
      6,
      12,
    )
    const result = detectDart(noisyReference, withDart, scene.calibration, config)
    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(describeScore(result.reading.score)).toBe('T20')
  })

  it('rejects a hand reaching across the board', () => {
    const result = detectDart(background, drawHand(scene, background), scene.calibration, config)
    expect(result.ok).toBe(false)
    if (result.ok) return
    expect(['not-dart-shaped', 'too-large']).toContain(result.reason)
  })

  it('ignores a change entirely off the board', () => {
    const offBoard = cloneImage(background)
    for (let y = 4; y < 30; y++) {
      for (let x = 4; x < 12; x++) offBoard.data[y * offBoard.width + x] = 250
    }
    const result = detectDart(background, offBoard, scene.calibration, config)
    expect(result.ok).toBe(false)
    if (result.ok) return
    expect(result.reason).toBe('nothing-on-board')
  })

  it('finds the second dart against a board that already has one', () => {
    const first = frameWithDarts(scene, background, [{ radiusMm: 103, bearingDeg: 0 }])
    const second = frameWithDarts(scene, background, [
      { radiusMm: 103, bearingDeg: 0 },
      { radiusMm: 166, bearingDeg: 90 },
    ])
    const result = detectDart(first, second, scene.calibration, config)
    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(describeScore(result.reading.score)).toBe('D6')
  })

  it('works through a perspective view', () => {
    const angled = makeScene(640, 480, 0.22)
    const angledBackground = makeBackground(angled)
    const withDart = frameWithDarts(angled, angledBackground, [{ radiusMm: 103, bearingDeg: 0 }])
    const result = detectDart(angledBackground, withDart, angled.calibration, config)
    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(describeScore(result.reading.score)).toBe('T20')
  })
})

describe('ThrowWatcher', () => {
  const scene = makeScene()
  const background = makeBackground(scene)

  function watcher() {
    return new ThrowWatcher(scene.calibration, config)
  }

  it('adopts the first frame as the baseline without reporting anything', () => {
    expect(watcher().push(background)).toMatchObject({ phase: 'waiting' })
  })

  it('stays quiet while the scene is still', () => {
    const w = watcher()
    w.push(background)
    for (let i = 0; i < 10; i++) {
      const update = w.push(addNoise(background, 3, i))
      expect(update.reading).toBeUndefined()
      expect(update.phase).toBe('waiting')
    }
  })

  it('reports the dart once the scene settles again', () => {
    const w = watcher()
    w.push(background)

    // An arm swings through.
    expect(w.push(drawHand(scene, background)).phase).toBe('moving')

    // Then it is gone and a dart is in the board.
    const landed = frameWithDarts(scene, background, [{ radiusMm: 103, bearingDeg: 0 }])
    let reading
    for (let i = 0; i < config.stableFrames + 2 && !reading; i++) {
      reading = w.push(landed).reading
    }
    expect(reading).toBeDefined()
    expect(describeScore(reading!.score)).toBe('T20')
  })

  it('will not report the same dart twice', () => {
    const w = watcher()
    w.push(background)
    w.push(drawHand(scene, background))

    const landed = frameWithDarts(scene, background, [{ radiusMm: 103, bearingDeg: 0 }])
    const readings = []
    for (let i = 0; i < 15; i++) {
      const update = w.push(landed)
      if (update.reading) readings.push(update.reading)
    }
    expect(readings).toHaveLength(1)
  })

  it('scores three darts in sequence', () => {
    const w = watcher()
    w.push(background)

    const darts = [
      { radiusMm: 103, bearingDeg: 0 }, // T20
      { radiusMm: 130, bearingDeg: 90 }, // 6
      { radiusMm: 166, bearingDeg: 180 }, // D3
    ]
    const scores: string[] = []

    for (let i = 0; i < darts.length; i++) {
      const inBoard = darts.slice(0, i + 1)
      w.push(drawHand(scene, background))
      const frame = frameWithDarts(scene, background, inBoard)
      for (let f = 0; f < config.stableFrames + 2; f++) {
        const update = w.push(frame)
        if (update.reading) scores.push(describeScore(update.reading.score))
      }
    }

    expect(scores).toEqual(['T20', '6', 'D3'])
  })

  it('re-baselines after darts are pulled out', () => {
    const w = watcher()
    const withDarts = frameWithDarts(scene, background, [{ radiusMm: 103, bearingDeg: 0 }])
    w.reset(withDarts)
    expect(w.referenceFrame).toBe(withDarts)

    w.reset(background)
    expect(w.referenceFrame).toBe(background)
  })

  it('reports why a settled frame produced nothing', () => {
    const w = watcher()
    w.push(background)
    w.push(drawHand(scene, background))

    let failure
    for (let i = 0; i < config.stableFrames + 2 && !failure; i++) {
      failure = w.push(background).failure
    }
    expect(failure).toBe('no-change')
  })
})
