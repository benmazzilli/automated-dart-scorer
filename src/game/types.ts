import type { BoardScore } from './board'

export type PlayerId = string

/** Where a throw's score came from. */
export type ThrowSource = 'manual' | 'camera'

export interface Throw {
  score: BoardScore
  /**
   * Board-space position in mm, when known. The camera always supplies one;
   * manual entry supplies one only when the score was entered by tapping the
   * board diagram rather than the number pad.
   */
  position?: { xMm: number; yMm: number }
  source: ThrowSource
  /** Camera confidence in `[0, 1]`. Absent for manual entry. */
  confidence?: number
  /** True if a camera reading was subsequently corrected by hand. */
  corrected?: boolean
  timestamp: number
}

/** A completed turn (visit to the oche). */
export interface Turn {
  playerId: PlayerId
  /** 1-based round number. */
  round: number
  throws: Throw[]
  /** Points credited for this turn after mode rules (0 for a bust). */
  scored: number
  /** True if the turn was voided by mode rules, e.g. going below zero at 501. */
  busted: boolean
}

export type GameStatus = 'playing' | 'finished'

/**
 * A complete game state.
 *
 * `Data` carries the mode-specific part: remaining scores for x01, lives and
 * assigned numbers for Killer, and so on. Everything else is common.
 *
 * States are treated as **immutable**. `GameMode.applyThrow` returns a new
 * state rather than mutating, which is what makes undo a matter of keeping the
 * previous state around — essential when the camera misreads a throw.
 */
export interface GameState<Data = unknown> {
  modeId: string
  players: PlayerId[]
  /** Index into `players` of whoever is at the oche. */
  currentPlayerIndex: number
  /** 1-based; increments when play wraps back to the first active player. */
  round: number
  /** Throws taken so far in the in-progress turn. */
  currentTurn: Throw[]
  /** Every completed turn, oldest first. */
  history: Turn[]
  status: GameStatus
  winner: PlayerId | null
  /** Final placings, first to last, for modes where players drop out. */
  rankings: PlayerId[]
  data: Data
  /** Short note for the UI about what just happened, e.g. `BUST!`. */
  message?: string
}

/**
 * A game mode, expressed as pure functions over state.
 *
 * Implementations must not mutate the state they are given. Keeping these as
 * pure reducers gives undo, replay and testability for free.
 */
export interface GameMode<Config = unknown, Data = unknown> {
  readonly id: string
  readonly name: string
  /** One-line explanation shown on the mode picker. */
  readonly description: string
  /** Sensible defaults so a game can always be started without configuring. */
  defaultConfig(): Config

  createInitialState(players: PlayerId[], config: Config): GameState<Data>

  /** Apply one dart. Must return a new state. */
  applyThrow(state: GameState<Data>, thrown: Throw): GameState<Data>

  /**
   * What the player at the oche is aiming for, e.g. `Needs 32 (D16)` or
   * `Target: 14`. Shown prominently during play.
   */
  describeTarget(state: GameState<Data>): string

  /**
   * Each player's headline number for the scoreboard — remaining score at
   * x01, lives at Killer. Ordered to match `state.players`.
   */
  scoreboard(state: GameState<Data>): PlayerScoreboardEntry[]
}

export interface PlayerScoreboardEntry {
  playerId: PlayerId
  /** The big number, e.g. remaining score or lives. */
  primary: string
  /** Optional supporting detail, e.g. a checkout suggestion or assigned number. */
  secondary?: string
  /** Rendered dimmed, e.g. a player knocked out of Killer. */
  eliminated: boolean
}

/** Maximum darts in a turn. Every mode we support uses three. */
export const DARTS_PER_TURN = 3
