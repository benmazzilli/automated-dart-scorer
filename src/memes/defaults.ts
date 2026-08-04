import type { MemeDefinition } from './triggers'

/**
 * The effects that ship with the app.
 *
 * All original — animations, colour and text only, no third-party assets.
 * Anything here can be overridden or added to by dropping a
 * `public/memes/memes.config.json` in place; see `loader.ts`.
 */
export const DEFAULT_MEMES: MemeDefinition[] = [
  {
    id: 'one-eighty',
    trigger: { on: 'turnScore', op: 'eq', value: 180 },
    text: 'ONE HUNDRED AND EIGHTYYY',
    effect: 'slam',
    colour: '#fbbf24',
    durationMs: 2600,
    priority: 100,
  },
  {
    id: 'nine-darter',
    trigger: { on: 'nineDarter' },
    text: 'NINE DARTER',
    effect: 'confetti',
    colour: '#f472b6',
    durationMs: 4000,
    priority: 200,
  },
  {
    id: 'big-checkout',
    trigger: { on: 'checkout', minValue: 100 },
    text: 'WHAT A FINISH',
    effect: 'confetti',
    colour: '#4ade80',
    durationMs: 2600,
    priority: 90,
  },
  {
    id: 'checkout',
    trigger: { on: 'checkout' },
    text: 'GAME SHOT',
    effect: 'confetti',
    colour: '#4ade80',
    durationMs: 2000,
    priority: 50,
  },
  {
    id: 'ton-forty',
    trigger: { on: 'turnScore', op: 'gte', value: 140 },
    text: 'ONE HUNDRED AND FORTY',
    effect: 'slam',
    colour: '#38bdf8',
    durationMs: 1800,
    priority: 60,
  },
  {
    id: 'ton',
    trigger: { on: 'turnScore', op: 'gte', value: 100 },
    text: 'TON',
    effect: 'slam',
    colour: '#38bdf8',
    durationMs: 1400,
    priority: 40,
  },
  {
    id: 'bag-o-nuts',
    trigger: { on: 'turnScore', op: 'eq', value: 26 },
    text: 'BAG O’ NUTS',
    effect: 'shake',
    colour: '#a3a3a3',
    durationMs: 1600,
    priority: 70,
  },
  {
    id: 'bust',
    trigger: { on: 'bust' },
    text: 'BUST',
    effect: 'shake',
    colour: '#f87171',
    durationMs: 1600,
    priority: 80,
  },
  {
    id: 'shocker',
    trigger: { on: 'turnScore', op: 'lte', value: 9 },
    text: 'OH DEAR',
    effect: 'flash',
    colour: '#f87171',
    durationMs: 1400,
    priority: 30,
  },
  {
    id: 'whitewash',
    trigger: { on: 'whitewash' },
    text: 'WHITEWASH',
    effect: 'confetti',
    colour: '#fbbf24',
    durationMs: 3500,
    priority: 210,
  },
]
