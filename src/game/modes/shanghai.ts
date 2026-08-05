/**
 * Shanghai.
 *
 * Round N is played on the number N and nothing else. Highest total after the
 * last round wins — unless someone throws a "Shanghai", a single, a double and
 * a treble of the round number in one visit, which wins instantly.
 */

import {
  commitTurn,
  createBaseState,
  currentPlayer,
  finishGame,
  isTurnComplete,
  sumThrows,
  withThrow,
} from '../engine'
import {
  type GameMode,
  type GameState,
  type PlayerId,
  type PlayerScoreboardEntry,
  type Throw,
} from '../types'

export interface ShanghaiConfig {
  /** Number of rounds. 7 is the usual short game; 20 goes all the way round. */
  rounds: number
  /** Whether a single, double and treble of the number wins outright. */
  instantWin: boolean
}

export interface ShanghaiData {
  config: ShanghaiConfig
  scores: Record<PlayerId, number>
}

/** The number in play for a given round. */
export function targetForRound(round: number): number {
  return round
}

/** Whether a set of darts is a Shanghai on the round's number. */
export function isShanghai(throws: readonly Throw[], target: number): boolean {
  const hits = throws.filter((t) => t.score.base === target && t.score.region !== 'miss')
  const multipliers = new Set(hits.map((t) => t.score.multiplier))
  return multipliers.has(1) && multipliers.has(2) && multipliers.has(3)
}

export const shanghaiMode: GameMode<ShanghaiConfig, ShanghaiData> = {
  id: 'shanghai',
  name: 'Shanghai',
  description: 'Round 1 is on the 1, round 2 on the 2, and so on. Single, double and treble wins outright.',

  defaultConfig(): ShanghaiConfig {
    return { rounds: 7, instantWin: true }
  },

  createInitialState(players: PlayerId[], config: ShanghaiConfig): GameState<ShanghaiData> {
    if (config.rounds < 1 || config.rounds > 20) throw new Error('Shanghai runs for 1 to 20 rounds')
    return createBaseState<ShanghaiData>('shanghai', players, {
      config,
      scores: Object.fromEntries(players.map((id) => [id, 0])),
    })
  },

  applyThrow(state: GameState<ShanghaiData>, thrown: Throw): GameState<ShanghaiData> {
    if (state.status === 'finished') return state

    const next = withThrow(state, thrown)
    const player = currentPlayer(next)
    const { config } = next.data
    const target = targetForRound(next.round)

    if (config.instantWin && isShanghai(next.currentTurn, target)) {
      const scored = onTargetTotal(next.currentTurn, target)
      const data: ShanghaiData = {
        ...next.data,
        scores: { ...next.data.scores, [player]: (next.data.scores[player] ?? 0) + scored },
      }
      return finishGame({ ...next, data }, player, `SHANGHAI! ${player} wins outright`)
    }

    if (!isTurnComplete(next)) return next

    const scored = onTargetTotal(next.currentTurn, target)
    const data: ShanghaiData = {
      ...next.data,
      scores: { ...next.data.scores, [player]: (next.data.scores[player] ?? 0) + scored },
    }

    const committed = commitTurn(next, { scored, data })

    // The round counter rolls past the last round once everyone has thrown.
    if (committed.round > config.rounds) {
      return finishGame(committed, highestScorer(committed), 'Highest score wins')
    }
    return committed
  },

  describeTarget(state: GameState<ShanghaiData>): string {
    if (state.status === 'finished') return 'Game over'
    const target = targetForRound(state.round)
    return `Round ${state.round} of ${state.data.config.rounds} — target ${target}`
  },

  scoreboard(state: GameState<ShanghaiData>): PlayerScoreboardEntry[] {
    return state.players.map((playerId) => ({
      playerId,
      primary: String(state.data.scores[playerId] ?? 0),
      secondary: undefined,
      eliminated: false,
    }))
  },
}

/** Total of the darts that landed on the round's number. */
function onTargetTotal(throws: readonly Throw[], target: number): number {
  return sumThrows(throws.filter((t) => t.score.base === target && t.score.region !== 'miss'))
}

/** The player with the highest score, or `null` on a tie. */
function highestScorer(state: GameState<ShanghaiData>): PlayerId | null {
  let best: PlayerId | null = null
  let bestScore = -1
  let tied = false
  for (const id of state.players) {
    const score = state.data.scores[id] ?? 0
    if (score > bestScore) {
      bestScore = score
      best = id
      tied = false
    } else if (score === bestScore) {
      tied = true
    }
  }
  return tied ? null : best
}
