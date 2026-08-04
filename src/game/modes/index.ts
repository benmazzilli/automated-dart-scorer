import type { GameMode } from '../types'
import { aroundTheClockMode } from './aroundTheClock'
import { halveItMode } from './halveIt'
import { killerMode } from './killer'
import { shanghaiMode } from './shanghai'
import { x01Mode } from './x01'

export { x01Mode, killerMode, aroundTheClockMode, shanghaiMode, halveItMode }

/**
 * Every playable mode, in the order the picker shows them.
 *
 * Modes are erased to `GameMode<any, any>` here on purpose: each carries its
 * own config and data shapes, and the UI holds them behind this common
 * interface. Anything needing the concrete types imports the mode directly.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const GAME_MODES: readonly GameMode<any, any>[] = [
  x01Mode,
  killerMode,
  aroundTheClockMode,
  shanghaiMode,
  halveItMode,
]

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function getMode(id: string): GameMode<any, any> {
  const mode = GAME_MODES.find((m) => m.id === id)
  if (!mode) throw new Error(`unknown game mode: ${id}`)
  return mode
}
