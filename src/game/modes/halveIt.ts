/**
 * Halve It.
 *
 * Each round sets a target. Score whatever you hit on that target — but miss
 * with all three darts and your score is halved. Cruel, and the reason it is
 * a good game with a drink in hand.
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

export type HalveItTarget =
  | { kind: 'number'; value: number }
  | { kind: 'anyDouble' }
  | { kind: 'anyTreble' }
  | { kind: 'bull' }

export interface HalveItConfig {
  targets: HalveItTarget[]
  /** Everyone starts here so there is something to lose in round one. */
  startingScore: number
}

export interface HalveItData {
  config: HalveItConfig
  scores: Record<PlayerId, number>
}

/** The classic round order. */
export const DEFAULT_TARGETS: HalveItTarget[] = [
  { kind: 'number', value: 20 },
  { kind: 'number', value: 19 },
  { kind: 'anyDouble' },
  { kind: 'number', value: 18 },
  { kind: 'number', value: 17 },
  { kind: 'anyTreble' },
  { kind: 'bull' },
]

export function describeTargetLabel(target: HalveItTarget): string {
  switch (target.kind) {
    case 'number':
      return String(target.value)
    case 'anyDouble':
      return 'any double'
    case 'anyTreble':
      return 'any treble'
    case 'bull':
      return 'bull'
  }
}

/** Whether a dart counts towards the round's target. */
export function hitsTarget(score: BoardScore, target: HalveItTarget): boolean {
  if (score.region === 'miss') return false
  switch (target.kind) {
    case 'number':
      return score.base === target.value
    case 'anyDouble':
      return score.region === 'double'
    case 'anyTreble':
      return score.region === 'treble'
    case 'bull':
      return score.region === 'inner_bull' || score.region === 'outer_bull'
  }
}

export const halveItMode: GameMode<HalveItConfig, HalveItData> = {
  id: 'halve-it',
  name: 'Halve It',
  description: 'Hit the round target or watch half your score disappear.',

  defaultConfig(): HalveItConfig {
    return { targets: DEFAULT_TARGETS, startingScore: 40 }
  },

  createInitialState(players: PlayerId[], config: HalveItConfig): GameState<HalveItData> {
    if (config.targets.length === 0) throw new Error('Halve It needs at least one target')
    return createBaseState<HalveItData>('halve-it', players, {
      config,
      scores: Object.fromEntries(players.map((id) => [id, config.startingScore])),
    })
  },

  applyThrow(state: GameState<HalveItData>, thrown: Throw): GameState<HalveItData> {
    if (state.status === 'finished') return state

    const next = withThrow(state, thrown)
    if (!isTurnComplete(next)) return next

    const player = currentPlayer(next)
    const { config } = next.data
    const target = config.targets[next.round - 1]
    if (target === undefined) return next

    const gained = next.currentTurn
      .filter((t) => hitsTarget(t.score, target))
      .reduce((total, t) => total + t.score.total, 0)

    const before = next.data.scores[player] ?? config.startingScore
    // Halving rounds down, so an odd score decays a little faster.
    const after = gained > 0 ? before + gained : Math.floor(before / 2)

    const data: HalveItData = { ...next.data, scores: { ...next.data.scores, [player]: after } }
    const message =
      gained > 0 ? undefined : `${player} missed ${describeTargetLabel(target)} — halved to ${after}`

    const committed = commitTurn(next, { scored: gained, data, message })

    if (committed.round > config.targets.length) {
      return finishGame(committed, highestScorer(committed), 'Highest score wins')
    }
    return committed
  },

  describeTarget(state: GameState<HalveItData>): string {
    if (state.status === 'finished') return 'Game over'
    const target = state.data.config.targets[state.round - 1]
    if (target === undefined) return 'Game over'
    return `Target: ${describeTargetLabel(target)}`
  },

  scoreboard(state: GameState<HalveItData>): PlayerScoreboardEntry[] {
    return state.players.map((playerId) => ({
      playerId,
      primary: String(state.data.scores[playerId] ?? state.data.config.startingScore),
      secondary: undefined,
      eliminated: false,
    }))
  },
}

function highestScorer(state: GameState<HalveItData>): PlayerId | null {
  let best: PlayerId | null = null
  let bestScore = -Infinity
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
