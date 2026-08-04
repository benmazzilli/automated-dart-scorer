import Dexie, { type Table } from 'dexie'
import type { MatchRecord } from '../game/stats'

export interface Profile {
  id: string
  name: string
  /** Accent colour used on the scoreboard, chosen at creation. */
  colour: string
  createdAt: number
}

/** Accent colours cycled through as profiles are added. */
export const PROFILE_COLOURS = [
  '#fbbf24',
  '#38bdf8',
  '#f87171',
  '#4ade80',
  '#c084fc',
  '#fb923c',
  '#2dd4bf',
  '#f472b6',
] as const

class OcheDatabase extends Dexie {
  profiles!: Table<Profile, string>
  matches!: Table<MatchRecord, string>

  constructor() {
    super('oche')
    this.version(1).stores({
      profiles: 'id, name, createdAt',
      // playerIds is multi-entry so a player's matches can be looked up
      // without scanning every record.
      matches: 'id, modeId, finishedAt, *playerIds',
    })
  }
}

export const db = new OcheDatabase()

export function newId(): string {
  return crypto.randomUUID()
}
