import { describe, it, expect } from 'vitest'
import { scoreFromSegment } from './board'
import {
  averageTrend,
  computeStats,
  formatAverage,
  formatRate,
  headToHead,
  type MatchRecord,
} from './stats'
import type { Throw, Turn } from './types'

function dart(base: number, multiplier: 1 | 2 | 3 = 1): Throw {
  return { score: scoreFromSegment(base, multiplier), source: 'manual', timestamp: 0 }
}

function turn(playerId: string, round: number, scored: number, darts = 3, busted = false): Turn {
  return {
    playerId,
    round,
    scored,
    busted,
    throws: Array.from({ length: darts }, () => dart(20)),
  }
}

function match(over: Partial<MatchRecord> = {}): MatchRecord {
  return {
    id: 'm1',
    modeId: 'x01',
    playerIds: ['alice', 'bob'],
    winnerId: 'alice',
    rankings: ['alice', 'bob'],
    turns: [],
    startingScore: 501,
    doubleOut: true,
    startedAt: 1000,
    finishedAt: 2000,
    ...over,
  }
}

describe('computeStats', () => {
  it('returns zeroes for a player with no matches', () => {
    const stats = computeStats('alice', [])
    expect(stats.matchesPlayed).toBe(0)
    expect(stats.threeDartAverage).toBe(0)
    expect(stats.bestLegDarts).toBeNull()
  })

  it('ignores matches the player was not in', () => {
    const stats = computeStats('carol', [match()])
    expect(stats.matchesPlayed).toBe(0)
  })

  it('counts matches played and won', () => {
    const stats = computeStats('alice', [match(), match({ id: 'm2', winnerId: 'bob' })])
    expect(stats.matchesPlayed).toBe(2)
    expect(stats.matchesWon).toBe(1)
    expect(stats.winRate).toBe(0.5)
  })

  it('computes a three-dart average', () => {
    const stats = computeStats('alice', [
      match({ turns: [turn('alice', 1, 60), turn('alice', 2, 90)] }),
    ])
    // 150 points from 6 darts is 75 per three darts.
    expect(stats.threeDartAverage).toBe(75)
    expect(stats.dartsThrown).toBe(6)
    expect(stats.totalScored).toBe(150)
  })

  it('counts big turns', () => {
    const stats = computeStats('alice', [
      match({
        turns: [
          turn('alice', 1, 180),
          turn('alice', 2, 140),
          turn('alice', 3, 100),
          turn('alice', 4, 99),
        ],
      }),
    ])
    expect(stats.count180).toBe(1)
    expect(stats.count140plus).toBe(2) // the 180 and the 140
    expect(stats.count100plus).toBe(3)
    expect(stats.highestTurn).toBe(180)
  })

  it('counts busts and excludes them from scoring', () => {
    const stats = computeStats('alice', [
      match({ turns: [turn('alice', 1, 60), turn('alice', 2, 0, 3, true)] }),
    ])
    expect(stats.busts).toBe(1)
    expect(stats.totalScored).toBe(60)
  })

  it('averages only the opening three visits for the first nine', () => {
    const stats = computeStats('alice', [
      match({
        turns: [
          turn('alice', 1, 180),
          turn('alice', 2, 180),
          turn('alice', 3, 180),
          turn('alice', 4, 0), // must not drag the first-nine figure down
        ],
      }),
    ])
    expect(stats.firstNineAverage).toBe(180)
    expect(stats.threeDartAverage).toBe(135) // 540 over 12 darts
  })

  it('records a checkout, the leg length and the highest finish', () => {
    const stats = computeStats('alice', [
      match({
        turns: [
          turn('alice', 1, 180),
          turn('alice', 2, 180),
          turn('alice', 3, 141), // 501 - 501 = 0
        ],
      }),
    ])
    expect(stats.checkoutsHit).toBe(1)
    expect(stats.highestCheckout).toBe(141)
    expect(stats.bestLegDarts).toBe(9)
  })

  it('counts a checkout attempt only when a finish was on', () => {
    const stats = computeStats('alice', [
      match({
        turns: [
          turn('alice', 1, 180), // from 501 — no finish on
          turn('alice', 2, 180), // from 321 — no finish on
          turn('alice', 3, 0), // from 141 — a finish was on, and missed
          turn('alice', 4, 141), // from 141 — on, and hit
        ],
      }),
    ])
    expect(stats.checkoutAttempts).toBe(2)
    expect(stats.checkoutsHit).toBe(1)
    expect(stats.checkoutRate).toBe(0.5)
  })

  it('resets the replay between legs', () => {
    // Alice wins leg one in nine darts, then plays a slower second leg.
    const stats = computeStats('alice', [
      match({
        turns: [
          turn('alice', 1, 180),
          turn('alice', 2, 180),
          turn('alice', 3, 141),
          turn('alice', 1, 100), // leg two, round counter restarts
          turn('alice', 2, 100),
        ],
      }),
    ])
    expect(stats.checkoutsHit).toBe(1)
    expect(stats.bestLegDarts).toBe(9)
    // The second leg's turns still count towards the average.
    expect(stats.dartsThrown).toBe(15)
  })

  it('resets the replay when an opponent wins a leg', () => {
    const stats = computeStats('alice', [
      match({
        turns: [
          turn('alice', 1, 100),
          turn('bob', 1, 180),
          turn('alice', 2, 100),
          turn('bob', 2, 180), // bob checks out; round restarts next
          turn('alice', 1, 60),
        ],
      }),
    ])
    // Alice's remaining must be back to 501 for the new leg, so her 60 from
    // 501 is not counted as a checkout attempt.
    expect(stats.checkoutAttempts).toBe(0)
    expect(stats.turns).toBe(3)
  })

  it('only scores x01 matches but counts other modes as played', () => {
    const stats = computeStats('alice', [
      match({ modeId: 'killer', turns: [turn('alice', 1, 0)] }),
    ])
    expect(stats.matchesPlayed).toBe(1)
    expect(stats.matchesWon).toBe(1)
    expect(stats.dartsThrown).toBe(0)
    expect(stats.threeDartAverage).toBe(0)
  })

  it('aggregates across several matches', () => {
    const stats = computeStats('alice', [
      match({ turns: [turn('alice', 1, 60)] }),
      match({ id: 'm2', turns: [turn('alice', 1, 120)] }),
    ])
    expect(stats.matchesPlayed).toBe(2)
    expect(stats.totalScored).toBe(180)
    expect(stats.threeDartAverage).toBe(90)
  })
})

describe('headToHead', () => {
  it('builds a record against each opponent', () => {
    const table = headToHead('alice', [
      match({ winnerId: 'alice' }),
      match({ id: 'm2', winnerId: 'bob' }),
      match({ id: 'm3', playerIds: ['alice', 'carol'], winnerId: 'alice' }),
    ])
    expect(table).toEqual([
      { opponentId: 'bob', played: 2, won: 1, lost: 1 },
      { opponentId: 'carol', played: 1, won: 1, lost: 0 },
    ])
  })

  it('skips drawn matches', () => {
    expect(headToHead('alice', [match({ winnerId: null })])).toEqual([])
  })

  it('handles a three-player game', () => {
    const table = headToHead('alice', [
      match({ playerIds: ['alice', 'bob', 'carol'], winnerId: 'bob' }),
    ])
    expect(table).toHaveLength(2)
    expect(table.every((row) => row.played === 1 && row.won === 0)).toBe(true)
    expect(table.find((r) => r.opponentId === 'bob')?.lost).toBe(1)
    // Carol did not beat Alice; neither of them won.
    expect(table.find((r) => r.opponentId === 'carol')?.lost).toBe(0)
  })
})

describe('averageTrend', () => {
  it('returns one point per x01 match, oldest first', () => {
    const trend = averageTrend('alice', [
      match({ id: 'b', finishedAt: 3000, turns: [turn('alice', 1, 120)] }),
      match({ id: 'a', finishedAt: 2000, turns: [turn('alice', 1, 60)] }),
    ])
    expect(trend.map((p) => p.at)).toEqual([2000, 3000])
    expect(trend.map((p) => p.average)).toEqual([60, 120])
  })

  it('drops matches with no scoring', () => {
    expect(averageTrend('alice', [match({ turns: [] })])).toEqual([])
  })
})

describe('formatting', () => {
  it('shows averages to one decimal place', () => {
    expect(formatAverage(84.333)).toBe('84.3')
    expect(formatAverage(0)).toBe('0.0')
  })

  it('shows rates as whole percentages', () => {
    expect(formatRate(0.5)).toBe('50%')
    expect(formatRate(0.333)).toBe('33%')
    expect(formatRate(0)).toBe('0%')
  })
})
