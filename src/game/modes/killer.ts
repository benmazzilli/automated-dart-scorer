/**
 * Killer.
 *
 * Every player owns a number. Hit your own double to become a killer, then
 * take lives off everyone else by hitting *their* double. Last one standing
 * wins.
 *
 * House rules vary wildly, so the fiddly bits are configurable: whether
 * hitting your own double once you are already a killer costs you a life, and
 * whether a treble takes extra lives.
 */

import { isDouble, SEGMENTS, type BoardScore } from '../board'
import {
  commitTurn,
  createBaseState,
  currentPlayer,
  eliminatePlayer,
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

export interface KillerConfig {
  /** Lives each player starts with. */
  lives: number
  /**
   * Whether a killer who hits their own double loses a life. The common house
   * rule, and it keeps killers honest.
   */
  selfHitCosts: boolean
  /** Whether a treble of a player's number takes three lives instead of one. */
  treblesCount: boolean
  /** Numbers assigned to players, in player order. Assigned randomly if absent. */
  numbers?: number[]
}

export interface KillerData {
  config: KillerConfig
  /** Each player's own number. */
  numbers: Record<PlayerId, number>
  lives: Record<PlayerId, number>
  /** Whether a player has hit their double and become a killer. */
  isKiller: Record<PlayerId, boolean>
}

/** Pick distinct numbers for each player, avoiding duplicates. */
function assignNumbers(players: readonly PlayerId[], preset?: number[]): Record<PlayerId, number> {
  if (preset) {
    if (preset.length < players.length) {
      throw new Error('not enough numbers supplied for the players in the game')
    }
    if (new Set(preset.slice(0, players.length)).size !== players.length) {
      throw new Error('each player needs a different number')
    }
    return Object.fromEntries(players.map((id, i) => [id, preset[i]!]))
  }

  if (players.length > SEGMENTS.length) {
    throw new Error(`Killer supports at most ${SEGMENTS.length} players`)
  }
  const pool = [...SEGMENTS]
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[pool[i], pool[j]] = [pool[j]!, pool[i]!]
  }
  return Object.fromEntries(players.map((id, i) => [id, pool[i]!]))
}

/**
 * Whether a dart on a player's number counts at all.
 *
 * Normally only the double does. With the treble house rule switched on a
 * treble counts too — for becoming a killer as well as for taking lives, so
 * the two stay consistent.
 */
function countsAsHit(score: BoardScore, config: KillerConfig): boolean {
  return isDouble(score) || (config.treblesCount && score.region === 'treble')
}

/** Lives removed by a dart landing on a given player's number. */
function livesTaken(score: BoardScore, config: KillerConfig): number {
  if (!countsAsHit(score, config)) return 0
  return config.treblesCount && score.region === 'treble' ? 3 : 1
}

export const killerMode: GameMode<KillerConfig, KillerData> = {
  id: 'killer',
  name: 'Killer',
  description: 'Hit your own double to become a killer, then knock the lives off everyone else.',

  defaultConfig(): KillerConfig {
    return { lives: 3, selfHitCosts: true, treblesCount: false }
  },

  createInitialState(players: PlayerId[], config: KillerConfig): GameState<KillerData> {
    if (players.length < 2) throw new Error('Killer needs at least two players')
    const numbers = assignNumbers(players, config.numbers)
    return createBaseState<KillerData>('killer', players, {
      config,
      numbers,
      lives: Object.fromEntries(players.map((id) => [id, config.lives])),
      isKiller: Object.fromEntries(players.map((id) => [id, false])),
    })
  },

  applyThrow(state: GameState<KillerData>, thrown: Throw): GameState<KillerData> {
    if (state.status === 'finished') return state

    const next = withThrow(state, thrown)
    const thrower = currentPlayer(next)
    const { config } = next.data

    let lives = { ...next.data.lives }
    let isKiller = { ...next.data.isKiller }
    let message: string | undefined

    // Which player owns the number that was hit? A dart on a number nobody owns
    // does nothing at all, which is most of the board.
    const owner = next.players.find((id) => next.data.numbers[id] === thrown.score.base)

    if (owner !== undefined && countsAsHit(thrown.score, config)) {
      if (owner === thrower) {
        if (!isKiller[thrower]) {
          isKiller = { ...isKiller, [thrower]: true }
          message = `${thrower} is a KILLER`
        } else if (config.selfHitCosts) {
          lives = { ...lives, [thrower]: Math.max(0, (lives[thrower] ?? 0) - 1) }
          message = `${thrower} hit their own number and lost a life`
        }
      } else if (isKiller[thrower]) {
        const taken = livesTaken(thrown.score, config)
        const before = lives[owner] ?? 0
        lives = { ...lives, [owner]: Math.max(0, before - taken) }
        message = `${thrower} took ${taken === 1 ? 'a life' : `${taken} lives`} off ${owner}`
      }
      // A non-killer hitting someone else's double does nothing.
    }

    const data: KillerData = { ...next.data, lives, isKiller }
    const isAlive = (id: PlayerId) => (data.lives[id] ?? 0) > 0

    // Bank anyone who has just been knocked out, so the finishing order is kept.
    let withEliminations: GameState<KillerData> = { ...next, data, message }
    for (const id of next.players) {
      if (!isAlive(id) && !withEliminations.rankings.includes(id)) {
        withEliminations = eliminatePlayer(withEliminations, id)
      }
    }

    const survivors = next.players.filter(isAlive)
    if (survivors.length <= 1) {
      return finishGame(withEliminations, survivors[0] ?? null, message)
    }

    if (isTurnComplete(withEliminations)) {
      return commitTurn(withEliminations, { scored: 0, data, isActive: isAlive, message })
    }

    // A player who runs out of lives mid-turn stops throwing immediately.
    if (!isAlive(thrower)) {
      return commitTurn(withEliminations, { scored: 0, data, isActive: isAlive, message })
    }

    return withEliminations
  },

  describeTarget(state: GameState<KillerData>): string {
    if (state.status === 'finished') return 'Game over'
    const player = currentPlayer(state)
    const own = state.data.numbers[player]
    if (!state.data.isKiller[player]) return `Hit D${own} to become a killer`
    const targets = state.players
      .filter((id) => id !== player && (state.data.lives[id] ?? 0) > 0)
      .map((id) => `D${state.data.numbers[id]}`)
    return `KILLER — hit ${targets.join(' or ')}`
  },

  scoreboard(state: GameState<KillerData>): PlayerScoreboardEntry[] {
    return state.players.map((playerId) => {
      const lives = state.data.lives[playerId] ?? 0
      return {
        playerId,
        primary: '♥'.repeat(lives) || 'OUT',
        secondary: `${state.data.isKiller[playerId] ? 'KILLER · ' : ''}number ${state.data.numbers[playerId]}`,
        eliminated: lives === 0,
      }
    })
  },
}
