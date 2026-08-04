/**
 * x01 — 501, 301 and friends.
 *
 * Players race from a starting score down to exactly zero. The wrinkles that
 * make this more than subtraction are bust handling, the double-in and
 * double-out rules, and leg/set structure.
 */

import { isDouble, type BoardScore } from '../board'
import { describeCheckout, findCheckout } from '../checkout'
import {
  commitTurn,
  createBaseState,
  currentPlayer,
  finishGame,
  isTurnComplete,
  withThrow,
} from '../engine'
import {
  DARTS_PER_TURN,
  type GameMode,
  type GameState,
  type PlayerId,
  type PlayerScoreboardEntry,
  type Throw,
} from '../types'

export interface X01Config {
  /** Usually 501 or 301, but any positive number works. */
  startingScore: number
  /** Require a double to start scoring. */
  doubleIn: boolean
  /** Require a double to finish. */
  doubleOut: boolean
  /** Legs needed to take a set (or to win outright when `setsToWin` is 1). */
  legsToWin: number
  /** Sets needed to win the match. Use 1 for a straight race of legs. */
  setsToWin: number
}

export interface X01Data {
  config: X01Config
  /** Score left at the **start of the current turn**, per player. */
  remaining: Record<PlayerId, number>
  /** Whether a player has opened, under double-in. Always true otherwise. */
  opened: Record<PlayerId, boolean>
  legsWon: Record<PlayerId, number>
  setsWon: Record<PlayerId, number>
  /** 1-based leg number within the current set. */
  leg: number
  /** 1-based set number. */
  set: number
  /** Index of whoever throws first this leg; rotates so the advantage shares. */
  startingPlayerIndex: number
}

export interface TurnOutcome {
  /** Score left after the darts so far, ignoring a bust. */
  remaining: number
  /** Points that will be credited, 0 if busted. */
  scored: number
  busted: boolean
  /** True when the player has checked out. */
  finished: boolean
  opened: boolean
  /** Which dart caused a bust or the checkout, for the message. */
  reason?: string
}

/**
 * Play a sequence of darts against a starting score and report what happened.
 *
 * Kept separate from state handling so the rules can be tested directly, and
 * so the UI can preview a turn in progress without committing it.
 */
export function evaluateTurn(
  config: X01Config,
  startingRemaining: number,
  alreadyOpened: boolean,
  throws: readonly Throw[],
): TurnOutcome {
  let remaining = startingRemaining
  let opened = alreadyOpened
  let scored = 0

  for (const thrown of throws) {
    const score: BoardScore = thrown.score

    // Under double-in, darts before the opening double simply do not count.
    if (!opened) {
      if (!isDouble(score)) continue
      opened = true
    }

    const next = remaining - score.total

    if (next < 0) {
      return { remaining: startingRemaining, scored: 0, busted: true, finished: false, opened: alreadyOpened, reason: 'Went below zero' }
    }

    if (config.doubleOut) {
      if (next === 1) {
        return { remaining: startingRemaining, scored: 0, busted: true, finished: false, opened: alreadyOpened, reason: 'Left 1 — no double to finish' }
      }
      if (next === 0) {
        if (!isDouble(score)) {
          return { remaining: startingRemaining, scored: 0, busted: true, finished: false, opened: alreadyOpened, reason: 'Did not finish on a double' }
        }
        return { remaining: 0, scored: scored + score.total, busted: false, finished: true, opened, reason: 'Checkout' }
      }
    } else if (next === 0) {
      return { remaining: 0, scored: scored + score.total, busted: false, finished: true, opened, reason: 'Checkout' }
    }

    remaining = next
    scored += score.total
  }

  return { remaining, scored, busted: false, finished: false, opened }
}

function record<T>(players: readonly PlayerId[], value: T): Record<PlayerId, T> {
  return Object.fromEntries(players.map((id) => [id, value]))
}

/** Fresh scores for a new leg, leaving legs and sets won intact. */
function resetForNewLeg(data: X01Data, players: readonly PlayerId[]): X01Data {
  return {
    ...data,
    remaining: record(players, data.config.startingScore),
    opened: record(players, !data.config.doubleIn),
  }
}

export const x01Mode: GameMode<X01Config, X01Data> = {
  id: 'x01',
  name: '501 / 301',
  description: 'Race from 501 (or 301) down to exactly zero, finishing on a double.',

  defaultConfig(): X01Config {
    return { startingScore: 501, doubleIn: false, doubleOut: true, legsToWin: 3, setsToWin: 1 }
  },

  createInitialState(players: PlayerId[], config: X01Config): GameState<X01Data> {
    if (config.startingScore < 2) throw new Error('starting score must be at least 2')
    return createBaseState<X01Data>('x01', players, {
      config,
      remaining: record(players, config.startingScore),
      opened: record(players, !config.doubleIn),
      legsWon: record(players, 0),
      setsWon: record(players, 0),
      leg: 1,
      set: 1,
      startingPlayerIndex: 0,
    })
  },

  applyThrow(state: GameState<X01Data>, thrown: Throw): GameState<X01Data> {
    if (state.status === 'finished') return state

    const next = withThrow(state, thrown)
    const player = currentPlayer(next)
    const { config } = next.data
    const startingRemaining = next.data.remaining[player] ?? config.startingScore
    const alreadyOpened = next.data.opened[player] ?? !config.doubleIn

    const outcome = evaluateTurn(config, startingRemaining, alreadyOpened, next.currentTurn)

    if (outcome.finished) {
      return winLeg(next, player, outcome)
    }

    if (outcome.busted) {
      return commitTurn(next, {
        scored: 0,
        busted: true,
        data: { ...next.data, opened: { ...next.data.opened, [player]: outcome.opened } },
        message: `BUST — ${outcome.reason}`,
      })
    }

    if (isTurnComplete(next)) {
      return commitTurn(next, {
        scored: outcome.scored,
        data: {
          ...next.data,
          remaining: { ...next.data.remaining, [player]: outcome.remaining },
          opened: { ...next.data.opened, [player]: outcome.opened },
        },
      })
    }

    return next
  },

  describeTarget(state: GameState<X01Data>): string {
    if (state.status === 'finished') return 'Game over'
    const player = currentPlayer(state)
    const { config } = state.data
    const opened = state.data.opened[player] ?? !config.doubleIn
    if (!opened) return 'Double to start'

    const live = liveRemaining(state, player)
    const dartsLeft = DARTS_PER_TURN - state.currentTurn.length
    const route = describeCheckout(findCheckout(live, dartsLeft, config.doubleOut))
    // With no finish on, the scoreboard already shows the number in large
    // type; repeating it here just adds a second copy of the same figure.
    return route ? `${live} — ${route}` : ''
  },

  scoreboard(state: GameState<X01Data>): PlayerScoreboardEntry[] {
    const { config } = state.data
    return state.players.map((playerId) => {
      const remaining =
        playerId === currentPlayer(state) && state.status === 'playing'
          ? liveRemaining(state, playerId)
          : (state.data.remaining[playerId] ?? config.startingScore)
      const legs = state.data.legsWon[playerId] ?? 0
      const sets = state.data.setsWon[playerId] ?? 0
      // The checkout route belongs on the target line, not here — showing it
      // in both places puts the same suggestion on screen twice.
      const tally = config.setsToWin > 1 ? `${sets} sets · ${legs} legs` : `${legs} legs`
      return {
        playerId,
        primary: String(remaining),
        secondary: tally,
        eliminated: false,
      }
    })
  },
}

/**
 * Score left for a player right now, accounting for darts already thrown in
 * the in-progress turn. Reverts to the turn's starting score if those darts
 * have busted.
 */
export function liveRemaining(state: GameState<X01Data>, playerId: PlayerId): number {
  const { config } = state.data
  const start = state.data.remaining[playerId] ?? config.startingScore
  if (playerId !== currentPlayer(state) || state.status === 'finished') return start
  const opened = state.data.opened[playerId] ?? !config.doubleIn
  return evaluateTurn(config, start, opened, state.currentTurn).remaining
}

/** Handle a checkout: bank the leg, and the set and match if they are done. */
function winLeg(
  state: GameState<X01Data>,
  player: PlayerId,
  outcome: TurnOutcome,
): GameState<X01Data> {
  const { config } = state.data
  const legsWon = { ...state.data.legsWon, [player]: (state.data.legsWon[player] ?? 0) + 1 }
  const legsInSet = legsWon[player] ?? 0

  const tookSet = legsInSet >= config.legsToWin
  const setsWon = tookSet
    ? { ...state.data.setsWon, [player]: (state.data.setsWon[player] ?? 0) + 1 }
    : state.data.setsWon
  const wonMatch = (setsWon[player] ?? 0) >= config.setsToWin

  // Rotate who throws first so the advantage of starting is shared out.
  const startingPlayerIndex = (state.data.startingPlayerIndex + 1) % state.players.length

  let data: X01Data = {
    ...state.data,
    legsWon,
    setsWon,
    startingPlayerIndex,
    leg: tookSet ? 1 : state.data.leg + 1,
    set: tookSet ? state.data.set + 1 : state.data.set,
  }
  // Taking a set resets the leg tally for the next one.
  if (tookSet && !wonMatch) {
    data = { ...data, legsWon: record(state.players, 0) }
  }
  data = resetForNewLeg(data, state.players)

  const banked = commitTurn(
    { ...state, data },
    {
      scored: outcome.scored,
      data,
      message: tookSet ? `Set to ${player}` : `Leg to ${player}`,
    },
  )

  if (wonMatch) {
    return finishGame(banked, player, `${player} wins the match`)
  }

  // The next leg starts with the rotated player, not whoever follows in order.
  return { ...banked, currentPlayerIndex: startingPlayerIndex }
}
