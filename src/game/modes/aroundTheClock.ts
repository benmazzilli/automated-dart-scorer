/**
 * Around the Clock.
 *
 * Work through 1 to 20 in order, then the bull. First player home wins.
 * A quick warm-up game, and the friendliest one for mixed abilities.
 */

import { type BoardScore } from '../board'
import {
  commitTurn,
  createBaseState,
  currentPlayer,
  finishGame,
  isTurnComplete,
  withThrow,
} from '../engine'
import {
  type GameMode,
  type GameState,
  type PlayerId,
  type PlayerScoreboardEntry,
  type Throw,
} from '../types'

/** Target 21 means the bull; 22 means home. */
const BULL_TARGET = 21
const HOME = 22

export interface AroundTheClockConfig {
  /** Whether the bull must be hit after the 20. */
  includeBull: boolean
  /** Whether a double advances two numbers and a treble three. */
  multiplesAdvance: boolean
}

export interface AroundTheClockData {
  config: AroundTheClockConfig
  /** The number each player is currently chasing. */
  target: Record<PlayerId, number>
}

function finishTarget(config: AroundTheClockConfig): number {
  return config.includeBull ? HOME : BULL_TARGET
}

/** How far a dart advances a player, or 0 if it missed their target. */
function advanceBy(target: number, score: BoardScore, config: AroundTheClockConfig): number {
  const onTarget =
    target === BULL_TARGET
      ? score.region === 'inner_bull' || score.region === 'outer_bull'
      : score.base === target && score.region !== 'miss'
  if (!onTarget) return 0
  if (!config.multiplesAdvance || target === BULL_TARGET) return 1
  return score.multiplier
}

export const aroundTheClockMode: GameMode<AroundTheClockConfig, AroundTheClockData> = {
  id: 'around-the-clock',
  name: 'Around the Clock',
  description: 'Hit 1 through 20 in order, then the bull. First one home wins.',

  defaultConfig(): AroundTheClockConfig {
    return { includeBull: true, multiplesAdvance: false }
  },

  createInitialState(players: PlayerId[], config: AroundTheClockConfig): GameState<AroundTheClockData> {
    return createBaseState<AroundTheClockData>('around-the-clock', players, {
      config,
      target: Object.fromEntries(players.map((id) => [id, 1])),
    })
  },

  applyThrow(state: GameState<AroundTheClockData>, thrown: Throw): GameState<AroundTheClockData> {
    if (state.status === 'finished') return state

    const next = withThrow(state, thrown)
    const player = currentPlayer(next)
    const { config } = next.data
    const home = finishTarget(config)

    const current = next.data.target[player] ?? 1
    const step = advanceBy(current, thrown.score, config)
    const updated = Math.min(current + step, home)

    const data: AroundTheClockData = {
      ...next.data,
      target: { ...next.data.target, [player]: updated },
    }
    const message = step > 0 ? (updated >= home ? `${player} is home!` : undefined) : undefined
    const advanced: GameState<AroundTheClockData> = { ...next, data, message }

    if (updated >= home) {
      return finishGame(advanced, player, `${player} goes all the way round`)
    }

    if (isTurnComplete(advanced)) {
      return commitTurn(advanced, { scored: 0, data })
    }
    return advanced
  },

  describeTarget(state: GameState<AroundTheClockData>): string {
    if (state.status === 'finished') return 'Game over'
    const target = state.data.target[currentPlayer(state)] ?? 1
    return target === BULL_TARGET ? 'Target: BULL' : `Target: ${target}`
  },

  scoreboard(state: GameState<AroundTheClockData>): PlayerScoreboardEntry[] {
    const home = finishTarget(state.data.config)
    return state.players.map((playerId) => {
      const target = state.data.target[playerId] ?? 1
      return {
        playerId,
        primary: target >= home ? 'HOME' : target === BULL_TARGET ? 'BULL' : String(target),
        secondary: `${Math.min(target - 1, home - 1)} of ${home - 1}`,
        eliminated: false,
      }
    })
  },
}
