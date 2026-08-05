import { DARTS_PER_TURN, type GameState, type PlayerId, type Throw, type Turn } from './types'

/** The player currently at the oche. */
export function currentPlayer<D>(state: GameState<D>): PlayerId {
  const id = state.players[state.currentPlayerIndex]
  if (id === undefined) {
    throw new Error(`currentPlayerIndex ${state.currentPlayerIndex} is out of range`)
  }
  return id
}

/** Build the common part of a fresh game state. */
export function createBaseState<D>(modeId: string, players: PlayerId[], data: D): GameState<D> {
  if (players.length === 0) throw new Error('a game needs at least one player')
  if (new Set(players).size !== players.length) {
    throw new Error('a player cannot appear twice in the same game')
  }
  return {
    modeId,
    players: [...players],
    currentPlayerIndex: 0,
    round: 1,
    currentTurn: [],
    history: [],
    status: 'playing',
    winner: null,
    rankings: [],
    data,
  }
}

/** Add a dart to the in-progress turn without applying any mode rules. */
export function withThrow<D>(state: GameState<D>, thrown: Throw): GameState<D> {
  return { ...state, currentTurn: [...state.currentTurn, thrown] }
}

/** Whether the in-progress turn has used all three darts. */
export function isTurnComplete<D>(state: GameState<D>): boolean {
  return state.currentTurn.length >= DARTS_PER_TURN
}

export interface CommitTurnOptions<D> {
  /** Points credited for the turn. Defaults to the sum of the darts thrown. */
  scored?: number
  busted?: boolean
  /** Replacement mode-specific data to carry into the next turn. */
  data?: D
  /**
   * Players who can still take a turn. Defaults to everyone. Used by
   * elimination modes so play skips knocked-out players.
   */
  isActive?: (playerId: PlayerId) => boolean
  message?: string
}

/**
 * End the current turn: bank it into history and hand over to the next active
 * player, rolling the round number when play wraps.
 *
 * A finished game is returned untouched, so a mode that declares a winner
 * mid-turn does not then advance past them.
 */
export function commitTurn<D>(state: GameState<D>, options: CommitTurnOptions<D> = {}): GameState<D> {
  const {
    scored = state.currentTurn.reduce((total, t) => total + t.score.total, 0),
    busted = false,
    data = state.data,
    isActive = () => true,
    message,
  } = options

  const turn: Turn = {
    playerId: currentPlayer(state),
    round: state.round,
    throws: state.currentTurn,
    scored,
    busted,
  }

  const committed: GameState<D> = {
    ...state,
    history: [...state.history, turn],
    currentTurn: [],
    data,
    message,
  }

  if (committed.status === 'finished') return committed
  return advanceToNextPlayer(committed, isActive)
}

/**
 * Move play to the next active player, incrementing the round each time the
 * index wraps past the end of the player list.
 *
 * If nobody is active the state is returned unchanged rather than looping
 * forever; modes are responsible for ending the game in that case.
 */
export function advanceToNextPlayer<D>(
  state: GameState<D>,
  isActive: (playerId: PlayerId) => boolean = () => true,
): GameState<D> {
  const count = state.players.length
  let index = state.currentPlayerIndex
  let round = state.round

  for (let step = 0; step < count; step++) {
    index += 1
    if (index >= count) {
      index = 0
      round += 1
    }
    const candidate = state.players[index]
    if (candidate !== undefined && isActive(candidate)) {
      return { ...state, currentPlayerIndex: index, round }
    }
  }

  return state
}

/** Total scored by a player across every completed turn. */
export function totalScored<D>(state: GameState<D>, playerId: PlayerId): number {
  return state.history
    .filter((turn) => turn.playerId === playerId)
    .reduce((total, turn) => total + turn.scored, 0)
}

/** Darts thrown by a player, including the in-progress turn. */
export function dartsThrown<D>(state: GameState<D>, playerId: PlayerId): number {
  const completed = state.history
    .filter((turn) => turn.playerId === playerId)
    .reduce((total, turn) => total + turn.throws.length, 0)
  const inProgress = currentPlayer(state) === playerId ? state.currentTurn.length : 0
  return completed + inProgress
}

/** Turns a player has completed. */
export function turnsTaken<D>(state: GameState<D>, playerId: PlayerId): number {
  return state.history.filter((turn) => turn.playerId === playerId).length
}

/**
 * Knock a player out of an elimination mode.
 *
 * `rankings` is always ordered **best first**. A player who is eliminated has
 * outlasted everyone eliminated before them, so they go on the *front* of the
 * list. Doing it this way means {@link finishGame} only has to put the winner
 * at the head and the whole finishing order is already correct.
 */
export function eliminatePlayer<D>(state: GameState<D>, playerId: PlayerId): GameState<D> {
  if (state.rankings.includes(playerId)) return state
  return { ...state, rankings: [playerId, ...state.rankings] }
}

/**
 * Declare the game over.
 *
 * The winner goes to the front, and anyone never explicitly placed is appended
 * behind those who were, so a scoreboard can always show a full finishing
 * order. See {@link eliminatePlayer} for the ordering convention.
 */
export function finishGame<D>(
  state: GameState<D>,
  winner: PlayerId | null,
  message?: string,
): GameState<D> {
  const placed = new Set(state.rankings)
  const rankings = [...state.rankings]
  if (winner !== null && !placed.has(winner)) {
    rankings.unshift(winner)
    placed.add(winner)
  }
  for (const player of state.players) {
    if (!placed.has(player)) rankings.push(player)
  }
  return { ...state, status: 'finished', winner, rankings, currentTurn: [], message }
}

/** Sum of a set of darts. */
export function sumThrows(throws: readonly Throw[]): number {
  return throws.reduce((total, t) => total + t.score.total, 0)
}
