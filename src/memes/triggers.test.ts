import { describe, it, expect } from 'vitest'
import { scoreFromSegment } from '../game/board'
import { x01Mode, type X01Data } from '../game/modes/x01'
import type { GameState, Throw } from '../game/types'
import { DEFAULT_MEMES } from './defaults'
import { detectEvents, matches, pickMeme, type MemeDefinition, type MemeEvent } from './triggers'

function dart(base: number, multiplier: 1 | 2 | 3 = 1): Throw {
  return { score: scoreFromSegment(base, multiplier), source: 'manual', timestamp: 0 }
}
const MISS = dart(0)

function start(startingScore = 501, legsToWin = 1): GameState<X01Data> {
  return x01Mode.createInitialState(['alice', 'bob'], {
    ...x01Mode.defaultConfig(),
    startingScore,
    legsToWin,
  })
}

/** Apply darts one at a time, collecting the events each transition produces. */
function playCollecting(state: GameState<X01Data>, throws: Throw[]) {
  const events: MemeEvent[] = []
  let current = state
  for (const thrown of throws) {
    const next = x01Mode.applyThrow(current, thrown)
    events.push(...detectEvents(current, next))
    current = next
  }
  return { state: current, events }
}

describe('matches', () => {
  const event: MemeEvent = { kind: 'turnScore', playerId: 'alice', value: 100 }

  it('compares turn scores by operator', () => {
    expect(matches({ on: 'turnScore', op: 'eq', value: 100 }, event)).toBe(true)
    expect(matches({ on: 'turnScore', op: 'eq', value: 99 }, event)).toBe(false)
    expect(matches({ on: 'turnScore', op: 'gte', value: 100 }, event)).toBe(true)
    expect(matches({ on: 'turnScore', op: 'gte', value: 101 }, event)).toBe(false)
    expect(matches({ on: 'turnScore', op: 'lte', value: 100 }, event)).toBe(true)
    expect(matches({ on: 'turnScore', op: 'lte', value: 99 }, event)).toBe(false)
  })

  it('does not cross event kinds', () => {
    expect(matches({ on: 'bust' }, event)).toBe(false)
  })

  it('applies a minimum to checkouts', () => {
    const checkout: MemeEvent = { kind: 'checkout', playerId: 'alice', value: 120 }
    expect(matches({ on: 'checkout' }, checkout)).toBe(true)
    expect(matches({ on: 'checkout', minValue: 100 }, checkout)).toBe(true)
    expect(matches({ on: 'checkout', minValue: 121 }, checkout)).toBe(false)
  })
})

describe('pickMeme', () => {
  const memes: MemeDefinition[] = [
    { id: 'low', trigger: { on: 'turnScore', op: 'gte', value: 100 }, effect: 'none', priority: 1 },
    { id: 'high', trigger: { on: 'turnScore', op: 'gte', value: 100 }, effect: 'none', priority: 9 },
  ]

  it('takes the highest priority match', () => {
    expect(pickMeme(memes, { kind: 'turnScore', playerId: 'a', value: 140 })?.id).toBe('high')
  })

  it('returns null when nothing matches', () => {
    expect(pickMeme(memes, { kind: 'bust', playerId: 'a', value: 0 })).toBeNull()
  })
})

describe('detectEvents', () => {
  it('reports a completed visit with its score', () => {
    const { events } = playCollecting(start(), [dart(20, 3), dart(20, 3), dart(20, 3)])
    expect(events).toContainEqual({ kind: 'turnScore', playerId: 'alice', value: 180 })
  })

  it('reports nothing mid-visit', () => {
    const { events } = playCollecting(start(), [dart(20, 3)])
    expect(events).toEqual([])
  })

  it('reports a bust instead of a score', () => {
    const { events } = playCollecting(start(40), [dart(20, 3)])
    expect(events).toEqual([{ kind: 'bust', playerId: 'alice', value: 0 }])
  })

  it('reports a checkout with the value that was taken out', () => {
    // 100 as T20 then D20 — the finish has to land on a double.
    const { events } = playCollecting(start(100, 2), [dart(20, 3), dart(20, 2)])
    expect(events).toContainEqual({ kind: 'checkout', playerId: 'alice', value: 100 })
  })

  it('reports a nine-darter', () => {
    // 501 = 180 + 180 + 141, with bob blanking in between.
    const nine = [
      dart(20, 3),
      dart(20, 3),
      dart(20, 3),
      MISS,
      MISS,
      MISS,
      dart(20, 3),
      dart(20, 3),
      dart(20, 3),
      MISS,
      MISS,
      MISS,
      dart(20, 3),
      dart(15, 3),
      dart(18, 2),
    ]
    const { events } = playCollecting(start(501, 2), nine)
    expect(events.some((e) => e.kind === 'nineDarter' && e.playerId === 'alice')).toBe(true)
  })

  it('does not call a short leg a nine-darter', () => {
    // Nine darts, but from 100 — that is an ordinary leg, not a nine-darter.
    const { events } = playCollecting(start(100, 2), [
      dart(20, 1),
      MISS,
      MISS,
      MISS,
      MISS,
      MISS,
      dart(20, 2),
      MISS,
      MISS,
      MISS,
      MISS,
      MISS,
      dart(20, 2),
    ])
    expect(events.some((e) => e.kind === 'checkout')).toBe(true)
    expect(events.some((e) => e.kind === 'nineDarter')).toBe(false)
  })

  it('does not report a nine-darter when a 501 leg took longer', () => {
    // Alice needs four visits, so twelve darts.
    const { events } = playCollecting(start(501, 2), [
      dart(20, 3),
      dart(20, 3),
      dart(20, 3), // 321
      MISS,
      MISS,
      MISS,
      dart(20, 3),
      dart(20, 3),
      dart(20, 3), // 141
      MISS,
      MISS,
      MISS,
      dart(20, 3),
      dart(20, 3),
      MISS, // 21
      MISS,
      MISS,
      MISS,
      dart(1, 1),
      dart(10, 2), // 21 - 1 - 20 = 0
    ])
    expect(events.some((e) => e.kind === 'checkout')).toBe(true)
    expect(events.some((e) => e.kind === 'nineDarter')).toBe(false)
  })

  it('reports a missed finish when one was on and not taken', () => {
    const state = start(40, 2)
    const { events } = playCollecting(state, [MISS, MISS, MISS])
    expect(events).toContainEqual({ kind: 'missedDouble', playerId: 'alice', value: 40 })
  })

  it('does not report a missed finish when no finish was on', () => {
    const { events } = playCollecting(start(501, 2), [MISS, MISS, MISS])
    expect(events.some((e) => e.kind === 'missedDouble')).toBe(false)
  })

  it('reports the game being won', () => {
    const { events } = playCollecting(start(40, 1), [dart(20, 2)])
    expect(events.some((e) => e.kind === 'gameWon' && e.playerId === 'alice')).toBe(true)
  })

  it('reports a whitewash when the loser takes no legs', () => {
    const { events } = playCollecting(start(40, 2), [
      dart(20, 2), // alice takes leg 1
      MISS,
      MISS,
      MISS, // bob blanks
      dart(20, 2), // alice takes leg 2 and the match
    ])
    expect(events.some((e) => e.kind === 'whitewash')).toBe(true)
  })

  it('ignores a state reset to a new game', () => {
    const played = playCollecting(start(), [dart(20, 3), dart(20, 3), dart(20, 3)]).state
    expect(detectEvents(played, start())).toEqual([])
  })
})

describe('default memes', () => {
  it('have unique ids', () => {
    expect(new Set(DEFAULT_MEMES.map((m) => m.id)).size).toBe(DEFAULT_MEMES.length)
  })

  it('cover the moments that matter', () => {
    const fire = (kind: MemeEvent['kind'], value: number) =>
      pickMeme(DEFAULT_MEMES, { kind, playerId: 'a', value })

    expect(fire('turnScore', 180)?.id).toBe('one-eighty')
    expect(fire('turnScore', 140)?.id).toBe('ton-forty')
    expect(fire('turnScore', 100)?.id).toBe('ton')
    expect(fire('turnScore', 26)?.id).toBe('bag-o-nuts')
    expect(fire('turnScore', 5)?.id).toBe('shocker')
    expect(fire('bust', 0)?.id).toBe('bust')
    expect(fire('checkout', 40)?.id).toBe('checkout')
    expect(fire('checkout', 120)?.id).toBe('big-checkout')
    expect(fire('nineDarter', 141)?.id).toBe('nine-darter')
    expect(fire('whitewash', 3)?.id).toBe('whitewash')
  })

  it('leaves an ordinary visit alone', () => {
    expect(pickMeme(DEFAULT_MEMES, { kind: 'turnScore', playerId: 'a', value: 45 })).toBeNull()
  })

  it('prefers the more specific meme when several match', () => {
    // 180 also satisfies the 100+ and 140+ rules; the 180 must win.
    expect(pickMeme(DEFAULT_MEMES, { kind: 'turnScore', playerId: 'a', value: 180 })?.id).toBe(
      'one-eighty',
    )
    // A nine-darter is also a checkout; the rarer one wins.
    expect(pickMeme(DEFAULT_MEMES, { kind: 'nineDarter', playerId: 'a', value: 141 })?.id).toBe(
      'nine-darter',
    )
  })
})
