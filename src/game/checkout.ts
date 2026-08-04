/**
 * Checkout suggestions for x01.
 *
 * Routes are searched rather than hardcoded from a table, so the same code
 * handles double-out and straight-out, and any number of darts remaining.
 * Candidate routes are then ranked to prefer the ones players actually throw:
 * fewest darts, a comfortable finishing double, and treble 20 to set up.
 */

import {
  SEGMENTS,
  allPossibleThrows,
  describeScore,
  isDouble,
  type BoardScore,
} from './board'

/** The highest score that can be checked out with three darts (T20, T20, bull). */
export const MAX_CHECKOUT = 170

/**
 * Cost of finishing on each double, keyed by the double's total. Lower is
 * better.
 *
 * Deliberately **not** a rank order. Ranking doubles 0, 1, 2, 3… and scaling
 * makes the gap between two equally good doubles as large as the gap between a
 * comfortable setup and a terrible one, which is how a solver ends up
 * suggesting D14 to set up D16 for a 60 checkout. The real differences are
 * lumpy: D20 and D16 are both excellent and nothing separates them; the odd
 * doubles are all similarly awkward; the bull is its own category because
 * missing it leaves 25 or 50 rather than a halved double.
 */
const FINISH_COST = new Map<number, number>([
  // Excellent — big, and halve cleanly down the D16 path.
  [40, 0], // D20
  [32, 0], // D16
  // Good.
  [36, 4], // D18
  [24, 4], // D12
  [20, 4], // D10
  [16, 4], // D8
  // Workable, but small or on an awkward part of the board.
  [28, 10], // D14
  [8, 10], // D4
  [12, 10], // D6
  [4, 10], // D2
  // Odd-numbered doubles: miss and you are left on an odd score again.
  [38, 25], // D19
  [34, 25], // D17
  [30, 25], // D15
  [26, 25], // D13
  [22, 25], // D11
  [18, 25], // D9
  [14, 25], // D7
  [10, 25], // D5
  [6, 25], // D3
  [2, 25], // D1
  // The bull finishes, but missing it leaves no double at all.
  [50, 30],
])

/**
 * Cost of a setup dart, by **how hard it is to hit** rather than how much it
 * scores.
 *
 * This distinction matters. Costing setups by score says T8 is a better dart
 * than a single 20, so a solver asked to check out 60 suggests T8 into D18
 * instead of the 20, D20 that every player actually throws. What a setup dart
 * is really for is leaving the right number, and among darts that leave an
 * equally good double the easiest target wins. Singles are a large area;
 * trebles and doubles are thin bands.
 *
 * Where a checkout genuinely needs big scoring — anything over about 100 — the
 * arithmetic forces trebles regardless, so this never gets in the way.
 */
function setupPreference(score: BoardScore): number {
  switch (score.region) {
    case 'miss':
      return 1000
    case 'single':
      return 0
    case 'treble':
      return 2
    case 'outer_bull':
      return 3
    case 'inner_bull':
      return 6
    case 'double':
      // Throwing at a double to set up another double risks checking out early
      // or busting, and it is a thin target for no extra scoring.
      return 12
  }
}

function finishPreference(score: BoardScore): number {
  // Straight-out permits finishing on a single or treble; those are worse than
  // any double but must still be reachable.
  return FINISH_COST.get(score.total) ?? (score.region === 'double' ? 25 : 60)
}

/** Throws worth considering, ordered so good setups are tried first. */
const SCORING_THROWS: readonly BoardScore[] = allPossibleThrows()
  .filter((t) => t.region !== 'miss')
  .sort((a, b) => setupPreference(a) - setupPreference(b))

/** Valid finishing darts under each rule, best first. */
const DOUBLE_FINISHES: readonly BoardScore[] = SCORING_THROWS.filter(isDouble).slice().sort(
  (a, b) => finishPreference(a) - finishPreference(b),
)
const ANY_FINISHES: readonly BoardScore[] = SCORING_THROWS.slice().sort((a, b) => {
  // Straight-out still prefers a double, but will take a single or treble.
  const byDouble = Number(isDouble(b)) - Number(isDouble(a))
  return byDouble !== 0 ? byDouble : setupPreference(a) - setupPreference(b)
})

export type CheckoutRoute = BoardScore[]

const cache = new Map<string, CheckoutRoute | null>()

/**
 * Find the best route to check out, or `null` if there isn't one.
 *
 * @param remaining   Score left to throw at.
 * @param dartsLeft   Darts available this turn (1–3).
 * @param doubleOut   Whether the final dart must be a double.
 */
export function findCheckout(
  remaining: number,
  dartsLeft: number,
  doubleOut = true,
): CheckoutRoute | null {
  if (!Number.isInteger(remaining) || remaining <= 0) return null
  if (dartsLeft < 1 || dartsLeft > 3) return null
  if (remaining > MAX_CHECKOUT) return null

  const key = `${remaining}:${dartsLeft}:${doubleOut}`
  const cached = cache.get(key)
  if (cached !== undefined) return cached

  const finishes = doubleOut ? DOUBLE_FINISHES : ANY_FINISHES

  // Try to finish in as few darts as possible, so search by increasing length.
  for (let length = 1; length <= dartsLeft; length++) {
    const route = bestRouteOfLength(remaining, length, finishes, doubleOut)
    if (route) {
      cache.set(key, route)
      return route
    }
  }

  cache.set(key, null)
  return null
}

/**
 * Cost of a complete route. Lower is better.
 *
 * Taking the first workable setup dart is not good enough: the best setup in
 * isolation is always T20, which for 64 would leave D2 — a poor double that no
 * player would choose over T16 into D8. So whole routes are costed, weighting
 * the finishing double above the setup darts but not so far above that the
 * setup stops mattering.
 */
function routeCost(route: CheckoutRoute): number {
  const finish = route[route.length - 1]!
  const setups = route.slice(0, -1)
  const setupCost = setups.reduce((total, dart) => total + setupPreference(dart), 0)
  // Both costs are already on a comparable scale, so they simply add. A better
  // double wins on its own merits, and setup difficulty breaks ties between
  // doubles that are equally good.
  //
  // The final term is a tiebreak only, far too small to change which darts get
  // picked. It orders equal-cost routes so the biggest setup goes first, since
  // "T20 20 D20" is how a player throws 120 and "20 T20 D20" is not.
  const ordering = setups.reduce((total, dart, index) => total + index * dart.total, 0)
  return finishPreference(finish) + setupCost + ordering * 0.001
}

/** Enumerate every route of exactly `length` darts and return the cheapest. */
function bestRouteOfLength(
  remaining: number,
  length: number,
  finishes: readonly BoardScore[],
  doubleOut: boolean,
): CheckoutRoute | null {
  let best: CheckoutRoute | null = null
  let bestCost = Infinity

  const consider = (route: CheckoutRoute) => {
    const cost = routeCost(route)
    if (cost < bestCost) {
      bestCost = cost
      best = route
    }
  }

  const walk = (left: number, dartsLeft: number, prefix: BoardScore[]) => {
    if (dartsLeft === 1) {
      for (const finish of finishes) {
        if (finish.total === left) consider([...prefix, finish])
      }
      return
    }
    for (const setup of SCORING_THROWS) {
      const next = left - setup.total
      // Under double-out, leaving 1 is a dead end and leaving 0 has already
      // finished on a non-double. Straight-out only needs something left to hit.
      const floor = doubleOut ? 2 : 1
      if (next < floor || next > (dartsLeft - 1) * 60) continue
      walk(next, dartsLeft - 1, [...prefix, setup])
    }
  }

  walk(remaining, length, [])
  return best
}

/** A checkout route as a readable string, e.g. `T20 T20 D20`. */
export function describeCheckout(route: CheckoutRoute | null): string | null {
  return route ? route.map(describeScore).join(' ') : null
}

/**
 * Whether a score can still be checked out this turn — used to decide whether
 * to show a suggestion at all.
 */
export function isCheckoutPossible(remaining: number, dartsLeft: number, doubleOut = true): boolean {
  return findCheckout(remaining, dartsLeft, doubleOut) !== null
}

/**
 * Scores from which no player can ever finish with three darts, even though
 * they are at or below the 170 ceiling. Shown as "no finish" in the UI.
 */
export const IMPOSSIBLE_CHECKOUTS: readonly number[] = [169, 168, 166, 165, 163, 162, 159]

/** Every segment value, for building the manual entry keypad. */
export const KEYPAD_SEGMENTS: readonly number[] = [...SEGMENTS].sort((a, b) => a - b)
