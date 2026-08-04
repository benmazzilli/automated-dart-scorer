import { describe, it, expect } from 'vitest'
import { scoreFromSegment } from '../board'
import { currentPlayer } from '../engine'
import type { GameState, Throw } from '../types'
import { evaluateTurn, liveRemaining, x01Mode, type X01Config, type X01Data } from './x01'

/** Build a throw, e.g. `dart(20, 3)` for a treble 20. */
function dart(base: number, multiplier: 1 | 2 | 3 = 1): Throw {
  return { score: scoreFromSegment(base, multiplier), source: 'manual', timestamp: 0 }
}
const MISS = dart(0)
const BULL = dart(50)

function start(config: Partial<X01Config> = {}, players = ['alice', 'bob']) {
  return x01Mode.createInitialState(players, { ...x01Mode.defaultConfig(), ...config })
}

function play(state: GameState<X01Data>, ...throws: Throw[]): GameState<X01Data> {
  return throws.reduce((s, t) => x01Mode.applyThrow(s, t), state)
}

describe('scoring and turns', () => {
  it('starts both players on the configured score', () => {
    const state = start({ startingScore: 501 })
    expect(state.data.remaining).toEqual({ alice: 501, bob: 501 })
    expect(currentPlayer(state)).toBe('alice')
  })

  it('subtracts a turn and hands over', () => {
    const state = play(start(), dart(20, 3), dart(20, 3), dart(20, 3))
    expect(state.data.remaining['alice']).toBe(321)
    expect(currentPlayer(state)).toBe('bob')
    expect(state.history).toHaveLength(1)
    expect(state.history[0]).toMatchObject({ playerId: 'alice', scored: 180, busted: false })
  })

  it('does not hand over mid-turn', () => {
    const state = play(start(), dart(20, 3), dart(20, 3))
    expect(currentPlayer(state)).toBe('alice')
    expect(state.currentTurn).toHaveLength(2)
    // Banked score only updates at the end of the turn...
    expect(state.data.remaining['alice']).toBe(501)
    // ...but the live figure reflects darts already thrown.
    expect(liveRemaining(state, 'alice')).toBe(381)
  })

  it('counts a miss as zero without ending the turn early', () => {
    const state = play(start(), MISS, MISS, MISS)
    expect(state.data.remaining['alice']).toBe(501)
    expect(state.history[0]).toMatchObject({ scored: 0, busted: false })
  })

  it('rolls the round number once play wraps', () => {
    let state = start()
    expect(state.round).toBe(1)
    state = play(state, dart(1), dart(1), dart(1)) // alice
    expect(state.round).toBe(1)
    state = play(state, dart(1), dart(1), dart(1)) // bob
    expect(state.round).toBe(2)
  })
})

describe('bust rules', () => {
  it('restores the score after a bust rather than leaving it mangled', () => {
    let state = start({ startingScore: 60 })
    state = play(state, dart(20, 3)) // lands on exactly 0, but T20 is no double
    expect(state.history[0]?.busted).toBe(true)
    expect(state.data.remaining['alice']).toBe(60)
  })

  it('busts when a dart takes the score below zero', () => {
    let state = start({ startingScore: 40 })
    state = play(state, dart(20, 3))
    expect(state.history[0]).toMatchObject({ busted: true, scored: 0 })
    expect(state.data.remaining['alice']).toBe(40)
    expect(state.message).toMatch(/below zero/i)
  })

  it('busts on leaving exactly 1 under double-out', () => {
    let state = start({ startingScore: 20 })
    state = play(state, dart(19))
    expect(state.history[0]?.busted).toBe(true)
    expect(state.message).toMatch(/no double/i)
  })

  it('busts on reaching zero with a single', () => {
    let state = start({ startingScore: 20 })
    state = play(state, dart(20, 1))
    expect(state.history[0]?.busted).toBe(true)
    expect(state.message).toMatch(/double/i)
  })

  it('discards the whole turn on a bust, including earlier good darts', () => {
    let state = start({ startingScore: 100 })
    // 20 leaves 80, another 20 leaves 60, then T20 would go to 0 on a single.
    state = play(state, dart(20), dart(20), dart(20, 3))
    expect(state.history[0]).toMatchObject({ busted: true, scored: 0 })
    expect(state.data.remaining['alice']).toBe(100)
  })

  it('ends the turn immediately on a bust without needing three darts', () => {
    let state = start({ startingScore: 40 })
    state = play(state, dart(20, 3))
    expect(currentPlayer(state)).toBe('bob')
    expect(state.currentTurn).toHaveLength(0)
  })

  it('does not bust on leaving 1 under straight-out', () => {
    let state = start({ startingScore: 20, doubleOut: false })
    state = play(state, dart(19))
    expect(state.history).toHaveLength(0) // turn continues
    expect(liveRemaining(state, 'alice')).toBe(1)
  })
})

describe('checkouts', () => {
  it('wins the leg by finishing on a double', () => {
    let state = start({ startingScore: 40, legsToWin: 1 })
    state = play(state, dart(20, 2))
    expect(state.status).toBe('finished')
    expect(state.winner).toBe('alice')
  })

  it('accepts the bull as a double finish', () => {
    let state = start({ startingScore: 50, legsToWin: 1 })
    state = play(state, BULL)
    expect(state.winner).toBe('alice')
  })

  it('does not accept the outer bull as a double finish', () => {
    let state = start({ startingScore: 25 })
    state = play(state, dart(25))
    expect(state.history[0]?.busted).toBe(true)
  })

  it('finishes on any dart under straight-out', () => {
    let state = start({ startingScore: 20, doubleOut: false, legsToWin: 1 })
    state = play(state, dart(20, 1))
    expect(state.winner).toBe('alice')
  })

  it('checks out mid-turn without throwing the remaining darts', () => {
    let state = start({ startingScore: 100, legsToWin: 1 })
    state = play(state, dart(20, 3), dart(20, 2))
    expect(state.winner).toBe('alice')
    expect(state.history[0]?.throws).toHaveLength(2)
  })
})

describe('double-in', () => {
  it('ignores darts before the opening double', () => {
    let state = start({ doubleIn: true, startingScore: 501 })
    state = play(state, dart(20, 3), dart(20, 1), dart(20, 3))
    expect(state.data.remaining['alice']).toBe(501)
    expect(state.data.opened['alice']).toBe(false)
    expect(state.history[0]?.scored).toBe(0)
  })

  it('starts scoring from the opening double onwards', () => {
    let state = start({ doubleIn: true, startingScore: 501 })
    state = play(state, dart(20, 1), dart(20, 2), dart(20, 3))
    // The single 20 does not count; D20 opens and scores, then T20 scores.
    expect(state.data.opened['alice']).toBe(true)
    expect(state.data.remaining['alice']).toBe(501 - 40 - 60)
    expect(state.history[0]?.scored).toBe(100)
  })

  it('stays open across turns', () => {
    let state = start({ doubleIn: true, startingScore: 501 })
    state = play(state, dart(20, 2), MISS, MISS) // alice opens
    state = play(state, MISS, MISS, MISS) // bob
    state = play(state, dart(20, 1), MISS, MISS) // alice scores normally now
    expect(state.data.remaining['alice']).toBe(501 - 40 - 20)
  })

  it('reports the target as needing a double to start', () => {
    const state = start({ doubleIn: true })
    expect(x01Mode.describeTarget(state)).toMatch(/double to start/i)
  })
})

describe('legs, sets and rotation', () => {
  it('resets scores and rotates the thrower after a leg', () => {
    let state = start({ startingScore: 40, legsToWin: 2 })
    state = play(state, dart(20, 2))
    expect(state.status).toBe('playing')
    expect(state.data.legsWon).toEqual({ alice: 1, bob: 0 })
    expect(state.data.remaining).toEqual({ alice: 40, bob: 40 })
    // Bob throws first in the second leg.
    expect(currentPlayer(state)).toBe('bob')
    expect(state.data.leg).toBe(2)
  })

  it('wins the match once enough legs are taken', () => {
    let state = start({ startingScore: 40, legsToWin: 2 })
    state = play(state, dart(20, 2)) // alice takes leg 1, bob now throws
    state = play(state, MISS, MISS, MISS) // bob blanks
    state = play(state, dart(20, 2)) // alice takes leg 2
    expect(state.status).toBe('finished')
    expect(state.winner).toBe('alice')
    expect(state.rankings).toEqual(['alice', 'bob'])
  })

  it('tracks sets and resets the leg tally between them', () => {
    let state = start({ startingScore: 40, legsToWin: 1, setsToWin: 2 })
    state = play(state, dart(20, 2)) // alice takes set 1
    expect(state.data.setsWon).toEqual({ alice: 1, bob: 0 })
    expect(state.data.legsWon).toEqual({ alice: 0, bob: 0 })
    expect(state.data.set).toBe(2)
    expect(state.status).toBe('playing')

    state = play(state, MISS, MISS, MISS) // bob throws first in set 2
    state = play(state, dart(20, 2)) // alice takes set 2 and the match
    expect(state.status).toBe('finished')
    expect(state.winner).toBe('alice')
  })

  it('ignores throws once the match is over', () => {
    let state = start({ startingScore: 40, legsToWin: 1 })
    state = play(state, dart(20, 2))
    const finished = state
    state = play(state, dart(20, 3))
    expect(state).toBe(finished)
  })
})

describe('target and scoreboard', () => {
  it('suggests a checkout when one is available', () => {
    const state = start({ startingScore: 40 })
    expect(x01Mode.describeTarget(state)).toBe('40 — D20')
  })

  it('says nothing when no checkout is on', () => {
    // The scoreboard already shows the remaining score in large type, so
    // repeating it on the target line would put the same figure on screen
    // twice. An empty target hides the line entirely.
    const state = start({ startingScore: 501 })
    expect(x01Mode.describeTarget(state)).toBe('')
  })

  it('keeps the checkout route off the scoreboard', () => {
    const state = start({ startingScore: 40 })
    expect(x01Mode.scoreboard(state)[0]?.secondary).toBe('0 legs')
  })

  it('narrows the suggestion as darts are used', () => {
    let state = start({ startingScore: 100 })
    expect(x01Mode.describeTarget(state)).toBe('100 — T20 D20')
    state = play(state, dart(20, 3))
    expect(x01Mode.describeTarget(state)).toBe('40 — D20')
  })

  it('reports live scores for the player at the oche', () => {
    let state = start({ startingScore: 501 })
    state = play(state, dart(20, 3))
    const board = x01Mode.scoreboard(state)
    expect(board.find((e) => e.playerId === 'alice')?.primary).toBe('441')
    expect(board.find((e) => e.playerId === 'bob')?.primary).toBe('501')
  })
})

describe('evaluateTurn', () => {
  const config: X01Config = {
    startingScore: 501,
    doubleIn: false,
    doubleOut: true,
    legsToWin: 1,
    setsToWin: 1,
  }

  it('is pure and handles an empty turn', () => {
    expect(evaluateTurn(config, 501, true, [])).toMatchObject({ remaining: 501, scored: 0, busted: false, finished: false })
  })

  it('reports the checkout dart', () => {
    expect(evaluateTurn(config, 32, true, [dart(16, 2)])).toMatchObject({ finished: true, remaining: 0, scored: 32 })
  })

  it('reports a bust reason', () => {
    expect(evaluateTurn(config, 20, true, [dart(19)]).reason).toMatch(/no double/i)
    expect(evaluateTurn(config, 10, true, [dart(20)]).reason).toMatch(/below zero/i)
  })
})
