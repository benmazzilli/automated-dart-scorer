import { DEFAULT_MEMES } from './defaults'
import type { MemeDefinition } from './triggers'

/** Where user-supplied images and sounds live, relative to the site root. */
export const MEME_ASSET_BASE = '/memes/'

interface MemeConfigFile {
  /** Replace the built-in set entirely rather than adding to it. */
  replaceDefaults?: boolean
  memes: MemeDefinition[]
}

/** Resolve an asset name in the config to a URL the browser can load. */
export function assetUrl(name: string): string {
  if (/^(https?:)?\/\//.test(name) || name.startsWith('/')) return name
  return MEME_ASSET_BASE + name
}

function isValid(meme: unknown): meme is MemeDefinition {
  if (typeof meme !== 'object' || meme === null) return false
  const candidate = meme as Partial<MemeDefinition>
  return typeof candidate.id === 'string' && typeof candidate.trigger === 'object'
}

/**
 * Load the meme set: the built-in effects, plus anything in
 * `public/memes/memes.config.json`.
 *
 * The config is fetched at runtime rather than imported, so new memes can be
 * dropped in without rebuilding the app. A missing or broken file is not an
 * error — it just means the built-in effects are used, which is the state
 * every install starts in.
 */
export async function loadMemes(signal?: AbortSignal): Promise<MemeDefinition[]> {
  try {
    const response = await fetch(`${MEME_ASSET_BASE}memes.config.json`, {
      ...(signal ? { signal } : {}),
      cache: 'no-cache',
    })
    if (!response.ok) return DEFAULT_MEMES

    const parsed = (await response.json()) as MemeConfigFile
    const custom = Array.isArray(parsed.memes) ? parsed.memes.filter(isValid) : []
    if (custom.length === 0) return DEFAULT_MEMES
    if (parsed.replaceDefaults) return custom

    // Custom definitions override built-ins sharing an id.
    const overridden = new Set(custom.map((m) => m.id))
    return [...DEFAULT_MEMES.filter((m) => !overridden.has(m.id)), ...custom]
  } catch {
    return DEFAULT_MEMES
  }
}
