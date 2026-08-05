/**
 * Statistics, computed from stored match history.
 *
 * Matches are persisted as their full turn list rather than as pre-aggregated
 * numbers, so a new statistic can be added later and immediately apply to
 * every game already played. The cost is recomputation, which for a few
 * hundred matches of sixty turns each is nothing.
 */

import type { PlayerId, Turn } from './types'
import { DARTS_PER_TURN } from './types'

export interface MatchRecord {
  id: string
  modeId: string
  playerIds: PlayerId[]
  winnerId: PlayerId | null
  rankings: PlayerId[]
  turns: Turn[]
  /** Present for x01 matches; needed to replay remaining scores. */
  startingScore?: number
  doubleOut?: boolean
  startedAt: number
  finishedAt: number
}

export interface PlayerStats {
  playerId: PlayerId
  matchesPlayed: number
  matchesWon: number
  winRate: number

  dartsThrown: number
  totalScored: number
  /** Points per three darts. The headline number in darts. */
  threeDartAverage: number
  /** Average over each leg's opening nine darts, where form shows first. */
  firstNineAverage: number

  turns: number
  busts: number
  highestTurn: number
  count180: number
  count140plus: number
  count100plus: number

  /** Visits where a checkout was on. */
  checkoutAttempts: number
  checkoutsHit: number
  checkoutRate: number
  highestCheckout: number
  /** Fewest darts used to win a leg. */
  bestLegDarts: number | null
}

export function emptyStats(playerId: PlayerId): PlayerStats {
  return {
    playerId,
    matchesPlayed: 0,
    matchesWon: 0,
    winRate: 0,
    dartsThrown: 0,
    totalScored: 0,
    threeDartAverage: 0,
    firstNineAverage: 0,
    turns: 0,
    busts: 0,
    highestTurn: 0,
    count180: 0,
    count140plus: 0,
    count100plus: 0,
    checkoutAttempts: 0,
    checkoutsHit: 0,
    checkoutRate: 0,
    highestCheckout: 0,
    bestLegDarts: null,
  }
}

/** The largest score that can be checked out in three darts. */
const MAX_CHECKOUT = 170

/**
 * Aggregate a player's statistics across a set of matches.
 *
 * x01 matches contribute scoring statistics; other modes contribute only
 * played and won, since a 3-dart average means nothing in Killer.
 */
export function computeStats(playerId: PlayerId, matches: readonly MatchRecord[]): PlayerStats {
  const stats = emptyStats(playerId)
  let firstNinePoints = 0
  let firstNineDarts = 0

  for (const match of matches) {
    if (!match.playerIds.includes(playerId)) continue

    stats.matchesPlayed += 1
    if (match.winnerId === playerId) stats.matchesWon += 1

    if (match.modeId !== 'x01') continue

    const startingScore = match.startingScore ?? 501

    // Replay the match leg by leg so remaining scores — and therefore whether
    // a checkout was on — can be reconstructed from the turn list alone.
    let remaining = startingScore
    let legDarts = 0
    let legTurns = 0

    for (const [index, turn] of match.turns.entries()) {
      const isOurs = turn.playerId === playerId

      if (isOurs) {
        stats.turns += 1
        stats.dartsThrown += turn.throws.length
        stats.totalScored += turn.scored
        if (turn.busted) stats.busts += 1
        if (turn.scored > stats.highestTurn) stats.highestTurn = turn.scored
        if (turn.scored === 180) stats.count180 += 1
        if (turn.scored >= 140) stats.count140plus += 1
        if (turn.scored >= 100) stats.count100plus += 1

        // The opening nine darts of a leg — three complete visits. Counting
        // whole visits keeps points and darts in step even when a visit ends
        // early on a checkout or a bust.
        if (legTurns < 3) {
          firstNinePoints += turn.scored
          firstNineDarts += turn.throws.length
        }

        // A checkout was on if the score at the start of the visit could be
        // finished at all.
        if (remaining >= 2 && remaining <= MAX_CHECKOUT) stats.checkoutAttempts += 1

        const before = remaining
        if (!turn.busted) remaining -= turn.scored
        legDarts += turn.throws.length
        legTurns += 1

        if (remaining === 0) {
          stats.checkoutsHit += 1
          if (before > stats.highestCheckout) stats.highestCheckout = before
          if (stats.bestLegDarts === null || legDarts < stats.bestLegDarts) {
            stats.bestLegDarts = legDarts
          }
        }
      }

      // A leg ends when we check out, or when the round number restarts —
      // which is how an opponent's checkout shows up in the turn list.
      if (remaining === 0 || startsNewLeg(match.turns[index + 1], turn)) {
        remaining = startingScore
        legDarts = 0
        legTurns = 0
      }
    }
  }

  stats.threeDartAverage =
    stats.dartsThrown > 0 ? (stats.totalScored / stats.dartsThrown) * DARTS_PER_TURN : 0
  stats.firstNineAverage =
    firstNineDarts > 0 ? (firstNinePoints / firstNineDarts) * DARTS_PER_TURN : 0
  stats.winRate = stats.matchesPlayed > 0 ? stats.matchesWon / stats.matchesPlayed : 0
  stats.checkoutRate =
    stats.checkoutAttempts > 0 ? stats.checkoutsHit / stats.checkoutAttempts : 0

  return stats
}

/**
 * Whether the next turn belongs to a new leg.
 *
 * Only this player's own arithmetic is replayed, so an opponent checking out
 * has to be spotted another way: the round counter restarts at 1 for the new
 * leg, and turns are stored in play order, which makes the drop reliable.
 */
function startsNewLeg(next: Turn | undefined, current: Turn): boolean {
  return next !== undefined && next.round < current.round
}

export interface HeadToHead {
  opponentId: PlayerId
  played: number
  won: number
  lost: number
}

/** Win/loss record against each other player met in the given matches. */
export function headToHead(playerId: PlayerId, matches: readonly MatchRecord[]): HeadToHead[] {
  const table = new Map<PlayerId, HeadToHead>()

  for (const match of matches) {
    if (!match.playerIds.includes(playerId)) continue
    if (match.winnerId === null) continue

    for (const opponentId of match.playerIds) {
      if (opponentId === playerId) continue
      const row = table.get(opponentId) ?? { opponentId, played: 0, won: 0, lost: 0 }
      row.played += 1
      if (match.winnerId === playerId) row.won += 1
      else if (match.winnerId === opponentId) row.lost += 1
      table.set(opponentId, row)
    }
  }

  return [...table.values()].sort((a, b) => b.played - a.played)
}

/**
 * Three-dart average per match over time, oldest first, for a trend line.
 * Only x01 matches produce a meaningful point.
 */
export function averageTrend(
  playerId: PlayerId,
  matches: readonly MatchRecord[],
): { at: number; average: number }[] {
  return matches
    .filter((m) => m.modeId === 'x01' && m.playerIds.includes(playerId))
    .sort((a, b) => a.finishedAt - b.finishedAt)
    .map((match) => ({
      at: match.finishedAt,
      average: computeStats(playerId, [match]).threeDartAverage,
    }))
    .filter((point) => point.average > 0)
}

/** Round to one decimal place for display. */
export function formatAverage(value: number): string {
  return value.toFixed(1)
}

/** Format a rate in `[0, 1]` as a whole-number percentage. */
export function formatRate(value: number): string {
  return `${Math.round(value * 100)}%`
}
