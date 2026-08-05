import type { GameState } from '../game/types'
import type { MatchRecord } from '../game/stats'
import type { X01Data } from '../game/modes/x01'
import { db, newId, PROFILE_COLOURS, type Profile } from './schema'

export async function listProfiles(): Promise<Profile[]> {
  return db.profiles.orderBy('createdAt').toArray()
}

export async function createProfile(name: string): Promise<Profile> {
  const trimmed = name.trim()
  if (!trimmed) throw new Error('a player needs a name')

  const existing = await db.profiles.where('name').equals(trimmed).first()
  if (existing) return existing

  const count = await db.profiles.count()
  const profile: Profile = {
    id: newId(),
    name: trimmed,
    colour: PROFILE_COLOURS[count % PROFILE_COLOURS.length]!,
    createdAt: Date.now(),
  }
  await db.profiles.add(profile)
  return profile
}

export async function renameProfile(id: string, name: string): Promise<void> {
  const trimmed = name.trim()
  if (!trimmed) throw new Error('a player needs a name')
  await db.profiles.update(id, { name: trimmed })
}

/**
 * Remove a profile. Their matches are kept, because deleting them would
 * rewrite the other players' head-to-head records and averages.
 */
export async function deleteProfile(id: string): Promise<void> {
  await db.profiles.delete(id)
}

export async function listMatches(playerId?: string): Promise<MatchRecord[]> {
  const matches = playerId
    ? await db.matches.where('playerIds').equals(playerId).toArray()
    : await db.matches.toArray()
  return matches.sort((a, b) => b.finishedAt - a.finishedAt)
}

/**
 * Persist a finished game.
 *
 * The full turn list is stored rather than pre-aggregated statistics, so new
 * statistics added later apply retrospectively to games already played.
 */
export async function saveMatch(
  state: GameState<unknown>,
  startedAt: number,
): Promise<MatchRecord | null> {
  if (state.status !== 'finished') return null

  const x01 = state.modeId === 'x01' ? (state.data as X01Data) : null

  const record: MatchRecord = {
    id: newId(),
    modeId: state.modeId,
    playerIds: [...state.players],
    winnerId: state.winner,
    rankings: [...state.rankings],
    turns: state.history,
    startedAt,
    finishedAt: Date.now(),
    ...(x01 ? { startingScore: x01.config.startingScore, doubleOut: x01.config.doubleOut } : {}),
  }

  await db.matches.add(record)
  return record
}

/** Wipe everything. Used by the settings screen and by tests. */
export async function clearAll(): Promise<void> {
  await db.transaction('rw', db.profiles, db.matches, async () => {
    await db.profiles.clear()
    await db.matches.clear()
  })
}
