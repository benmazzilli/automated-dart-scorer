/**
 * Detecting the moments worth reacting to, and matching them against a
 * declarative list of meme definitions.
 *
 * Events are derived by diffing two game states rather than by hooking into
 * each mode, so a new mode gets the meme system for free and the rules engine
 * stays free of presentation concerns.
 */

import type { X01Data } from '../game/modes/x01'
import type { GameState, PlayerId, Turn } from '../game/types'

export type MemeEventKind =
  | 'turnScore'
  | 'bust'
  | 'checkout'
  | 'missedDouble'
  | 'gameWon'
  | 'whitewash'
  | 'nineDarter'

export interface MemeEvent {
  kind: MemeEventKind
  playerId: PlayerId
  /** Turn total, or the checkout value for a finish. */
  value: number
}

export type MemeTrigger =
  | { on: 'turnScore'; op: 'eq' | 'gte' | 'lte'; value: number }
  | { on: 'bust' }
  | { on: 'checkout'; minValue?: number }
  | { on: 'missedDouble' }
  | { on: 'gameWon' }
  | { on: 'whitewash' }
  | { on: 'nineDarter' }

/** The built-in animations. Assets are optional extras on top. */
export type EffectStyle = 'slam' | 'shake' | 'confetti' | 'flash' | 'none'

export interface MemeDefinition {
  id: string
  trigger: MemeTrigger
  /** Big text across the screen. */
  text?: string
  effect: EffectStyle
  /** Accent colour for the text and flash. */
  colour?: string
  /** Optional user asset, resolved relative to the `memes/` folder. */
  image?: string
  sound?: string
  durationMs?: number
  /** Higher wins when several definitions match the same event. */
  priority?: number
}

/** Whether a trigger fires for an event. */
export function matches(trigger: MemeTrigger, event: MemeEvent): boolean {
  if (trigger.on !== event.kind) return false
  switch (trigger.on) {
    case 'turnScore':
      if (trigger.op === 'eq') return event.value === trigger.value
      if (trigger.op === 'gte') return event.value >= trigger.value
      return event.value <= trigger.value
    case 'checkout':
      return trigger.minValue === undefined || event.value >= trigger.minValue
    default:
      return true
  }
}

/** The highest-priority definition matching an event, if any. */
export function pickMeme(
  definitions: readonly MemeDefinition[],
  event: MemeEvent,
): MemeDefinition | null {
  let best: MemeDefinition | null = null
  for (const definition of definitions) {
    if (!matches(definition.trigger, event)) continue
    if (best === null || (definition.priority ?? 0) > (best.priority ?? 0)) best = definition
  }
  return best
}

/** Darts a player used in the leg that the given turn completed. */
function dartsInLeg(history: readonly Turn[], playerId: PlayerId): number {
  let darts = 0
  // Walk backwards to the start of the current leg, which is where the round
  // counter stops decreasing.
  for (let i = history.length - 1; i >= 0; i--) {
    const turn = history[i]!
    const previous = history[i - 1]
    if (turn.playerId === playerId) darts += turn.throws.length
    if (previous !== undefined && previous.round > turn.round) break
  }
  return darts
}

function x01Of(state: GameState<unknown>): X01Data | null {
  return state.modeId === 'x01' ? (state.data as X01Data) : null
}

/** Total legs won across all players, used to spot a leg changing hands. */
function legsWonTotal(data: X01Data | null): number {
  if (!data) return 0
  return Object.values(data.legsWon).reduce((total, n) => total + n, 0)
}

/**
 * Work out what just happened between two states.
 *
 * Only transitions that add a completed turn or finish the game produce
 * events, so this can be called on every state change cheaply.
 */
export function detectEvents(
  previous: GameState<unknown>,
  next: GameState<unknown>,
): MemeEvent[] {
  const events: MemeEvent[] = []

  const turnAdded = next.history.length > previous.history.length
  const newTurn = turnAdded ? next.history[next.history.length - 1] : undefined

  if (newTurn) {
    const previousX01 = x01Of(previous)
    const nextX01 = x01Of(next)
    const wonLeg = legsWonTotal(nextX01) > legsWonTotal(previousX01)

    if (newTurn.busted) {
      events.push({ kind: 'bust', playerId: newTurn.playerId, value: 0 })
    } else if (wonLeg && previousX01) {
      const checkoutValue = previousX01.remaining[newTurn.playerId] ?? newTurn.scored
      events.push({ kind: 'checkout', playerId: newTurn.playerId, value: checkoutValue })

      // A nine-darter is specifically a 501 leg in nine darts. Checking out a
      // short leg in nine is just an ordinary leg, so the starting score has
      // to be part of the test.
      if (
        previousX01.config.startingScore >= 501 &&
        dartsInLeg(next.history, newTurn.playerId) <= 9
      ) {
        events.push({ kind: 'nineDarter', playerId: newTurn.playerId, value: checkoutValue })
      }
    } else {
      events.push({ kind: 'turnScore', playerId: newTurn.playerId, value: newTurn.scored })

      // A finish was on at the start of the visit and was not taken.
      if (previousX01) {
        const before = previousX01.remaining[newTurn.playerId] ?? Infinity
        if (before >= 2 && before <= 170) {
          events.push({ kind: 'missedDouble', playerId: newTurn.playerId, value: before })
        }
      }
    }
  }

  if (previous.status !== 'finished' && next.status === 'finished' && next.winner) {
    events.push({ kind: 'gameWon', playerId: next.winner, value: 0 })

    const data = x01Of(next)
    if (data) {
      const losersLegs = next.players
        .filter((id) => id !== next.winner)
        .reduce((total, id) => total + (data.legsWon[id] ?? 0), 0)
      const winnerLegs = data.legsWon[next.winner] ?? 0
      if (winnerLegs >= 2 && losersLegs === 0) {
        events.push({ kind: 'whitewash', playerId: next.winner, value: winnerLegs })
      }
    }
  }

  return events
}
