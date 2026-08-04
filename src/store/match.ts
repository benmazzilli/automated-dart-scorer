import { create } from 'zustand'
import { saveMatch } from '../db/queries'
import type { Profile } from '../db/schema'
import type { BoardScore } from '../game/board'
import { getMode } from '../game/modes'
import type { GameMode, GameState, PlayerId, Throw, ThrowSource } from '../game/types'

/** How many states back undo can reach. A leg is far shorter than this. */
const UNDO_LIMIT = 400

export interface ThrowInput {
  score: BoardScore
  position?: { xMm: number; yMm: number }
  source?: ThrowSource
  confidence?: number
  corrected?: boolean
}

interface MatchStore {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  mode: GameMode<any, any> | null
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  state: GameState<any> | null
  /** Previous states, oldest first. Undo pops the last one. */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  past: GameState<any>[]
  /** Profiles of everyone in the current game, for resolving names and colours. */
  profiles: Profile[]
  /** When the current game began, for the stored match record. */
  startedAt: number

  startGame(modeId: string, profiles: Profile[], config?: unknown): void
  /** Display name for a player id. */
  nameOf(playerId: PlayerId): string
  colourOf(playerId: PlayerId): string
  throwDart(input: ThrowInput): void
  /** Step back one dart. */
  undo(): void
  /** Step back to the start of the current turn. */
  undoTurn(): void
  quit(): void

  canUndo(): boolean
}

export const useMatch = create<MatchStore>((set, get) => ({
  mode: null,
  state: null,
  past: [],
  profiles: [],
  startedAt: 0,

  startGame(modeId, profiles, config) {
    const mode = getMode(modeId)
    const resolved = config ?? mode.defaultConfig()
    set({
      mode,
      state: mode.createInitialState(
        profiles.map((p) => p.id),
        resolved,
      ),
      past: [],
      profiles,
      startedAt: Date.now(),
    })
  },

  nameOf(playerId) {
    return get().profiles.find((p) => p.id === playerId)?.name ?? playerId
  },

  colourOf(playerId) {
    return get().profiles.find((p) => p.id === playerId)?.colour ?? '#fbbf24'
  },

  throwDart(input) {
    const { mode, state, past } = get()
    if (!mode || !state || state.status === 'finished') return

    const thrown: Throw = {
      score: input.score,
      source: input.source ?? 'manual',
      timestamp: Date.now(),
      ...(input.position ? { position: input.position } : {}),
      ...(input.confidence !== undefined ? { confidence: input.confidence } : {}),
      ...(input.corrected ? { corrected: true } : {}),
    }

    const next = mode.applyThrow(state, thrown)
    // A mode that ignores a dart returns the same object; nothing to record.
    if (next === state) return

    set({ state: next, past: [...past, state].slice(-UNDO_LIMIT) })

    if (next.status === 'finished') {
      // Persist in the background. A failed write must not block the win
      // screen — the game has been played either way.
      void saveMatch(next, get().startedAt).catch((error: unknown) => {
        console.error('could not save match', error)
      })
    }
  },

  undo() {
    const { past } = get()
    const previous = past[past.length - 1]
    if (!previous) return
    set({ state: previous, past: past.slice(0, -1) })
  },

  undoTurn() {
    const { state, past } = get()
    if (!state) return

    let history = past
    let current = state

    const stepBack = (): boolean => {
      const previous = history[history.length - 1]
      if (!previous) return false
      current = previous
      history = history.slice(0, -1)
      return true
    }

    // Always give back at least one dart. Without this, calling undo-turn
    // straight after a turn was banked would do nothing at all, since the
    // in-progress turn is already empty — and that is exactly the moment a
    // player notices the camera scored the turn wrong.
    if (!stepBack()) return
    while (current.currentTurn.length > 0 && stepBack()) {
      // keep walking back to the first dart of that turn
    }

    set({ state: current, past: history })
  },

  quit() {
    set({ mode: null, state: null, past: [], profiles: [], startedAt: 0 })
  },

  canUndo() {
    return get().past.length > 0
  },
}))
