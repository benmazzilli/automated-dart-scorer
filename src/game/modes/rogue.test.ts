import { describe, it, expect } from 'vitest'
import { scoreFromSegment } from '../board'
import { currentPlayer } from '../engine'
import type { GameState, Throw } from '../types'
import { aroundTheClockMode } from './aroundTheClock'
import { halveItMode, hitsTarget, type HalveItConfig } from './halveIt'
import { killerMode, type KillerConfig } from './killer'
import { isShanghai, shanghaiMode, type ShanghaiConfig } from './shanghai'
import { GAME_MODES, getMode } from './index'

function dart(base: number, multiplier: 1 | 2 | 3 = 1): Throw {
  return { score: scoreFromSegment(base, multiplier), source: 'manual', timestamp: 0 }
}
const MISS = dart(0)

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function play<S extends GameState<any>>(mode: { applyThrow: (s: S, t: Throw) => S }, state: S, ...throws: Throw[]): S {
  return throws.reduce((s, t) => mode.applyThrow(s, t), state)
}

describe('Killer', () => {
  const config = (over: Partial<KillerConfig> = {}): KillerConfig => ({
    ...killerMode.defaultConfig(),
    numbers: [20, 19],
    ...over,
  })

  it('assigns the configured numbers', () => {
    const state = killerMode.createInitialState(['alice', 'bob'], config())
    expect(state.data.numbers).toEqual({ alice: 20, bob: 19 })
    expect(state.data.lives).toEqual({ alice: 3, bob: 3 })
    expect(state.data.isKiller).toEqual({ alice: false, bob: false })
  })

  it('assigns distinct random numbers when none are given', () => {
    const state = killerMode.createInitialState(
      ['a', 'b', 'c', 'd'],
      config({ numbers: undefined }),
    )
    const assigned = Object.values(state.data.numbers)
    expect(new Set(assigned).size).toBe(4)
  })

  it('rejects duplicate numbers', () => {
    expect(() =>
      killerMode.createInitialState(['alice', 'bob'], config({ numbers: [20, 20] })),
    ).toThrow(/different number/)
  })

  it('makes a player a killer when they hit their own double', () => {
    const state = play(killerMode, killerMode.createInitialState(['alice', 'bob'], config()), dart(20, 2))
    expect(state.data.isKiller['alice']).toBe(true)
    expect(state.message).toMatch(/KILLER/)
  })

  it('does nothing on a single of your own number', () => {
    const state = play(killerMode, killerMode.createInitialState(['alice', 'bob'], config()), dart(20, 1))
    expect(state.data.isKiller['alice']).toBe(false)
  })

  it('will not let a non-killer take lives', () => {
    const state = play(killerMode, killerMode.createInitialState(['alice', 'bob'], config()), dart(19, 2))
    expect(state.data.lives['bob']).toBe(3)
  })

  it('takes a life once a killer hits an opponent double', () => {
    let state = killerMode.createInitialState(['alice', 'bob'], config())
    state = play(killerMode, state, dart(20, 2), dart(19, 2))
    expect(state.data.lives['bob']).toBe(2)
    expect(state.message).toMatch(/took a life off bob/)
  })

  it('costs a killer a life for hitting their own number again', () => {
    let state = killerMode.createInitialState(['alice', 'bob'], config({ selfHitCosts: true }))
    state = play(killerMode, state, dart(20, 2), dart(20, 2))
    expect(state.data.lives['alice']).toBe(2)
  })

  it('leaves a killer alone on their own number when the house rule is off', () => {
    let state = killerMode.createInitialState(['alice', 'bob'], config({ selfHitCosts: false }))
    state = play(killerMode, state, dart(20, 2), dart(20, 2))
    expect(state.data.lives['alice']).toBe(3)
  })

  it('takes three lives on a treble when configured', () => {
    let state = killerMode.createInitialState(['alice', 'bob'], config({ treblesCount: true, lives: 5 }))
    state = play(killerMode, state, dart(20, 2), dart(19, 3))
    expect(state.data.lives['bob']).toBe(2)
  })

  it('wins when everyone else is knocked out', () => {
    let state = killerMode.createInitialState(['alice', 'bob'], config({ lives: 1 }))
    state = play(killerMode, state, dart(20, 2), dart(19, 2))
    expect(state.status).toBe('finished')
    expect(state.winner).toBe('alice')
    expect(state.rankings).toEqual(['alice', 'bob'])
  })

  it('ranks players by how long they lasted', () => {
    let state = killerMode.createInitialState(['alice', 'bob', 'carol'], {
      ...killerMode.defaultConfig(),
      numbers: [20, 19, 18],
      lives: 1,
    })
    state = play(killerMode, state, dart(20, 2), dart(19, 2), dart(18, 2))
    expect(state.status).toBe('finished')
    expect(state.winner).toBe('alice')
    // Carol was knocked out after Bob, so she placed above him.
    expect(state.rankings).toEqual(['alice', 'carol', 'bob'])
  })

  it('skips eliminated players when handing over', () => {
    let state = killerMode.createInitialState(['alice', 'bob', 'carol'], {
      ...killerMode.defaultConfig(),
      numbers: [20, 19, 18],
      lives: 1,
    })
    // Alice becomes a killer and knocks Bob out, then finishes her turn.
    state = play(killerMode, state, dart(20, 2), dart(19, 2), MISS)
    expect(state.data.lives['bob']).toBe(0)
    expect(state.status).toBe('playing')
    expect(currentPlayer(state)).toBe('carol')
  })

  it('describes the right target before and after becoming a killer', () => {
    let state = killerMode.createInitialState(['alice', 'bob'], config())
    expect(killerMode.describeTarget(state)).toBe('Hit D20 to become a killer')
    state = play(killerMode, state, dart(20, 2))
    expect(killerMode.describeTarget(state)).toBe('KILLER — hit D19')
  })

  it('needs at least two players', () => {
    expect(() => killerMode.createInitialState(['alice'], config())).toThrow(/two players/)
  })
})

describe('Around the Clock', () => {
  it('advances only on the current target', () => {
    let state = aroundTheClockMode.createInitialState(['alice'], aroundTheClockMode.defaultConfig())
    state = play(aroundTheClockMode, state, dart(5), dart(1))
    expect(state.data.target['alice']).toBe(2)
  })

  it('advances by the multiplier when configured', () => {
    let state = aroundTheClockMode.createInitialState(['alice'], {
      includeBull: true,
      multiplesAdvance: true,
    })
    state = play(aroundTheClockMode, state, dart(1, 3))
    expect(state.data.target['alice']).toBe(4)
  })

  it('accepts either bull for the final target', () => {
    let state = aroundTheClockMode.createInitialState(['alice'], {
      includeBull: true,
      multiplesAdvance: false,
    })
    state = { ...state, data: { ...state.data, target: { alice: 21 } } }
    state = play(aroundTheClockMode, state, dart(25))
    expect(state.status).toBe('finished')
    expect(state.winner).toBe('alice')
  })

  it('finishes on the 20 when the bull is excluded', () => {
    let state = aroundTheClockMode.createInitialState(['alice'], {
      includeBull: false,
      multiplesAdvance: false,
    })
    state = { ...state, data: { ...state.data, target: { alice: 20 } } }
    state = play(aroundTheClockMode, state, dart(20))
    expect(state.status).toBe('finished')
  })

  it('hands over after three darts', () => {
    let state = aroundTheClockMode.createInitialState(['alice', 'bob'], aroundTheClockMode.defaultConfig())
    state = play(aroundTheClockMode, state, MISS, MISS, MISS)
    expect(currentPlayer(state)).toBe('bob')
  })
})

describe('Shanghai', () => {
  const config = (over: Partial<ShanghaiConfig> = {}): ShanghaiConfig => ({
    ...shanghaiMode.defaultConfig(),
    ...over,
  })

  it('only scores the round number', () => {
    let state = shanghaiMode.createInitialState(['alice', 'bob'], config())
    state = play(shanghaiMode, state, dart(1), dart(20, 3), dart(1, 2))
    expect(state.data.scores['alice']).toBe(3) // 1 + 2, the T20 does not count
  })

  it('tracks the target by round', () => {
    let state = shanghaiMode.createInitialState(['alice', 'bob'], config())
    expect(shanghaiMode.describeTarget(state)).toMatch(/target 1$/)
    state = play(shanghaiMode, state, MISS, MISS, MISS)
    state = play(shanghaiMode, state, MISS, MISS, MISS)
    expect(shanghaiMode.describeTarget(state)).toMatch(/target 2$/)
  })

  it('wins outright on a Shanghai', () => {
    let state = shanghaiMode.createInitialState(['alice', 'bob'], config())
    state = play(shanghaiMode, state, dart(1, 1), dart(1, 2), dart(1, 3))
    expect(state.status).toBe('finished')
    expect(state.winner).toBe('alice')
    expect(state.message).toMatch(/SHANGHAI/)
  })

  it('does not win outright when the rule is off', () => {
    let state = shanghaiMode.createInitialState(['alice', 'bob'], config({ instantWin: false }))
    state = play(shanghaiMode, state, dart(1, 1), dart(1, 2), dart(1, 3))
    expect(state.status).toBe('playing')
    expect(state.data.scores['alice']).toBe(6)
  })

  it('ends after the configured rounds and the highest score wins', () => {
    let state = shanghaiMode.createInitialState(['alice', 'bob'], config({ rounds: 2 }))
    state = play(shanghaiMode, state, dart(1), MISS, MISS) // alice 1
    state = play(shanghaiMode, state, MISS, MISS, MISS) // bob 0
    state = play(shanghaiMode, state, MISS, MISS, MISS) // round 2, alice
    state = play(shanghaiMode, state, MISS, MISS, MISS) // round 2, bob
    expect(state.status).toBe('finished')
    expect(state.winner).toBe('alice')
  })

  it('records no winner on a tie', () => {
    let state = shanghaiMode.createInitialState(['alice', 'bob'], config({ rounds: 1 }))
    state = play(shanghaiMode, state, MISS, MISS, MISS)
    state = play(shanghaiMode, state, MISS, MISS, MISS)
    expect(state.status).toBe('finished')
    expect(state.winner).toBeNull()
  })

  it('detects a Shanghai only with all three multipliers', () => {
    expect(isShanghai([dart(5, 1), dart(5, 2), dart(5, 3)], 5)).toBe(true)
    expect(isShanghai([dart(5, 1), dart(5, 1), dart(5, 3)], 5)).toBe(false)
    expect(isShanghai([dart(5, 1), dart(5, 2), dart(6, 3)], 5)).toBe(false)
  })
})

describe('Halve It', () => {
  const config = (over: Partial<HalveItConfig> = {}): HalveItConfig => ({
    ...halveItMode.defaultConfig(),
    ...over,
  })

  it('adds what you hit on target', () => {
    let state = halveItMode.createInitialState(['alice', 'bob'], config())
    state = play(halveItMode, state, dart(20), dart(20, 3), MISS)
    expect(state.data.scores['alice']).toBe(40 + 20 + 60)
  })

  it('halves the score when all three darts miss the target', () => {
    let state = halveItMode.createInitialState(['alice', 'bob'], config({ startingScore: 41 }))
    state = play(halveItMode, state, dart(5), dart(7), dart(3))
    expect(state.data.scores['alice']).toBe(20) // rounds down
    expect(state.message).toMatch(/halved to 20/)
  })

  it('matches the special targets', () => {
    expect(hitsTarget(scoreFromSegment(11, 2), { kind: 'anyDouble' })).toBe(true)
    expect(hitsTarget(scoreFromSegment(11, 3), { kind: 'anyDouble' })).toBe(false)
    expect(hitsTarget(scoreFromSegment(4, 3), { kind: 'anyTreble' })).toBe(true)
    expect(hitsTarget(scoreFromSegment(50, 1), { kind: 'bull' })).toBe(true)
    expect(hitsTarget(scoreFromSegment(25, 1), { kind: 'bull' })).toBe(true)
    expect(hitsTarget(scoreFromSegment(20, 1), { kind: 'bull' })).toBe(false)
  })

  it('scores an any-double round with the double it hit', () => {
    let state = halveItMode.createInitialState(['alice', 'bob'], config())
    // Round 3 is "any double" in the default target list.
    state = { ...state, round: 3 }
    state = play(halveItMode, state, dart(19, 2), MISS, MISS)
    expect(state.data.scores['alice']).toBe(40 + 38)
  })

  it('ends after the last target and the highest score wins', () => {
    let state = halveItMode.createInitialState(['alice', 'bob'], config({ targets: [{ kind: 'number', value: 20 }] }))
    state = play(halveItMode, state, dart(20), MISS, MISS) // alice 60
    state = play(halveItMode, state, MISS, MISS, MISS) // bob halved to 20
    expect(state.status).toBe('finished')
    expect(state.winner).toBe('alice')
    expect(state.data.scores).toEqual({ alice: 60, bob: 20 })
  })

  it('names the target for each round', () => {
    const state = halveItMode.createInitialState(['alice'], config())
    expect(halveItMode.describeTarget(state)).toBe('Target: 20')
    expect(halveItMode.describeTarget({ ...state, round: 3 })).toBe('Target: any double')
    expect(halveItMode.describeTarget({ ...state, round: 7 })).toBe('Target: bull')
  })
})

describe('mode registry', () => {
  it('exposes all five modes with unique ids', () => {
    expect(GAME_MODES).toHaveLength(5)
    expect(new Set(GAME_MODES.map((m) => m.id)).size).toBe(5)
  })

  it('every mode can start a game and describe a target from its defaults', () => {
    for (const mode of GAME_MODES) {
      const state = mode.createInitialState(['alice', 'bob'], mode.defaultConfig())
      expect(state.status).toBe('playing')
      expect(state.players).toEqual(['alice', 'bob'])
      expect(typeof mode.describeTarget(state)).toBe('string')
      expect(mode.scoreboard(state)).toHaveLength(2)
    }
  })

  it('every mode survives a barrage of random darts without throwing', () => {
    // A crude fuzz: modes must never crash or hand over to a missing player,
    // whatever sequence of darts arrives. The camera will produce odd ones.
    const bases = [0, 1, 5, 20, 19, 18, 25, 50]
    for (const mode of GAME_MODES) {
      let state = mode.createInitialState(['alice', 'bob', 'carol'], mode.defaultConfig())
      for (let i = 0; i < 200 && state.status === 'playing'; i++) {
        const base = bases[i % bases.length]!
        const multiplier = ((i % 3) + 1) as 1 | 2 | 3
        state = mode.applyThrow(state, dart(base === 25 || base === 50 ? base : base, base > 20 ? 1 : multiplier))
        expect(state.players).toContain(state.players[state.currentPlayerIndex])
        expect(state.currentTurn.length).toBeLessThanOrEqual(3)
      }
    }
  })

  it('looks modes up by id', () => {
    expect(getMode('x01').name).toBe('501 / 301')
    expect(() => getMode('nope')).toThrow(/unknown game mode/)
  })
})
