import { describe, it, expect } from 'vitest'
import { describeScore, isDouble } from './board'
import {
  IMPOSSIBLE_CHECKOUTS,
  MAX_CHECKOUT,
  describeCheckout,
  findCheckout,
  isCheckoutPossible,
} from './checkout'

describe('findCheckout', () => {
  it('finds the maximum checkout', () => {
    expect(describeCheckout(findCheckout(MAX_CHECKOUT, 3))).toBe('T20 T20 BULL')
  })

  it('finishes on a double in one dart where possible', () => {
    expect(describeCheckout(findCheckout(40, 1))).toBe('D20')
    expect(describeCheckout(findCheckout(32, 1))).toBe('D16')
    expect(describeCheckout(findCheckout(50, 1))).toBe('BULL')
    expect(describeCheckout(findCheckout(2, 1))).toBe('D1')
  })

  it('cannot finish an odd number with one dart under double-out', () => {
    expect(findCheckout(41, 1)).toBeNull()
    expect(findCheckout(3, 1)).toBeNull()
    expect(findCheckout(1, 1)).toBeNull()
  })

  it('uses the conventional two-dart routes', () => {
    // Only scores with one unambiguous published route are pinned exactly.
    // Plenty of checkouts have several routes that different tables disagree
    // on, so those are asserted by property below rather than by string.
    expect(describeCheckout(findCheckout(100, 2))).toBe('T20 D20')
    expect(describeCheckout(findCheckout(60, 2))).toBe('20 D20')
    expect(describeCheckout(findCheckout(170, 3))).toBe('T20 T20 BULL')
  })

  it('breaks a tie between equally hard setups on the better double', () => {
    // 80 is T20 into D10 in some tables and T16 into D16 in others. Both setup
    // darts are trebles, so nothing separates them on difficulty and the
    // better double should decide it — D16 halves cleanly, D10 does not.
    expect(describeCheckout(findCheckout(80, 2))).toBe('T16 D16')
  })

  it('does not strand the player on an awkward double to use a bigger treble', () => {
    // The trap: T20 is the best opening dart in isolation, so a greedy search
    // checks 64 out as T20 D2. D2 is a rotten double to leave when T16 into D8
    // is available for the same two darts.
    const route = findCheckout(64, 2)!
    expect(route).toHaveLength(2)
    expect(route.reduce((sum, t) => sum + t.total, 0)).toBe(64)
    expect(isDouble(route[1]!)).toBe(true)
    expect(route[1]!.total).toBeGreaterThanOrEqual(8)
  })

  it('leaves a reasonable double on every two-dart checkout', () => {
    for (let remaining = 4; remaining <= 110; remaining++) {
      const route = findCheckout(remaining, 2)
      if (route === null || route.length !== 2) continue
      // D1 is only acceptable when arithmetic forces it.
      if (route[1]!.total === 2) {
        expect(findCheckout(remaining - 2, 1), `${remaining} could have avoided D1`).toBeNull()
      }
    }
  })

  it('always returns a route that sums correctly and ends on a double', () => {
    for (let remaining = 2; remaining <= MAX_CHECKOUT; remaining++) {
      const route = findCheckout(remaining, 3)
      if (route === null) continue
      const total = route.reduce((sum, t) => sum + t.total, 0)
      expect(total, `route for ${remaining}: ${describeCheckout(route)}`).toBe(remaining)
      expect(route.length).toBeLessThanOrEqual(3)
      expect(isDouble(route[route.length - 1]!), `${remaining} must end on a double`).toBe(true)
    }
  })

  it('identifies exactly the scores with no three-dart finish', () => {
    // Derived from the solver rather than asserted from a remembered table,
    // then compared against the published list.
    const noFinish: number[] = []
    for (let remaining = 2; remaining <= MAX_CHECKOUT; remaining++) {
      if (findCheckout(remaining, 3) === null) noFinish.push(remaining)
    }
    expect(noFinish.sort((a, b) => b - a)).toEqual([...IMPOSSIBLE_CHECKOUTS])
  })

  it('rejects scores above the ceiling and nonsense input', () => {
    expect(findCheckout(171, 3)).toBeNull()
    expect(findCheckout(180, 3)).toBeNull()
    expect(findCheckout(0, 3)).toBeNull()
    expect(findCheckout(-5, 3)).toBeNull()
    expect(findCheckout(40.5, 3)).toBeNull()
    expect(findCheckout(40, 0)).toBeNull()
    expect(findCheckout(40, 4)).toBeNull()
  })

  it('never leaves a score of 1 as an intermediate step', () => {
    // Leaving 1 is a dead end under double-out; a suggested route must never
    // walk a player into it.
    for (let remaining = 2; remaining <= MAX_CHECKOUT; remaining++) {
      const route = findCheckout(remaining, 3)
      if (route === null) continue
      let left = remaining
      for (const dart of route.slice(0, -1)) {
        left -= dart.total
        expect(left, `route for ${remaining} left 1`).toBeGreaterThanOrEqual(2)
      }
    }
  })

  it('respects the darts remaining in the turn', () => {
    // 170 needs all three darts, so it is not available with two left.
    expect(findCheckout(170, 3)).not.toBeNull()
    expect(findCheckout(170, 2)).toBeNull()
    expect(findCheckout(100, 2)).not.toBeNull()
    expect(findCheckout(100, 1)).toBeNull()
  })
})

describe('straight-out', () => {
  it('allows finishing on a single or treble', () => {
    expect(findCheckout(1, 1, false)).not.toBeNull()
    expect(describeScore(findCheckout(1, 1, false)![0]!)).toBe('1')
    expect(findCheckout(3, 1, false)).not.toBeNull()
  })

  it('can finish everything double-out can, and more', () => {
    // Dropping the double requirement can only ever add routes, never remove
    // them. A few high scores stay unreachable regardless: 163 for instance is
    // not the sum of any three throws, double or not.
    const unreachable: number[] = []
    for (let remaining = 1; remaining <= MAX_CHECKOUT; remaining++) {
      const straight = findCheckout(remaining, 3, false)
      if (straight === null) {
        unreachable.push(remaining)
        expect(findCheckout(remaining, 3, true), `${remaining} impossible straight-out but not double-out`).toBeNull()
      }
    }
    // Odd scores below 4 and a handful of high ones are simply not sums of
    // three darts. Pin the set so a regression in the throw table is caught.
    expect(unreachable).toEqual([163, 166, 169])
  })

  it('still prefers a double when one is available', () => {
    expect(isDouble(findCheckout(40, 1, false)![0]!)).toBe(true)
  })
})

describe('isCheckoutPossible', () => {
  it('agrees with findCheckout', () => {
    expect(isCheckoutPossible(170, 3)).toBe(true)
    expect(isCheckoutPossible(169, 3)).toBe(false)
    expect(isCheckoutPossible(501, 3)).toBe(false)
  })
})
