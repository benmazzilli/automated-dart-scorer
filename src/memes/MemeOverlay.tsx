import { useEffect, useRef, useState } from 'react'
import confetti from 'canvas-confetti'
import { useMatch } from '../store/match'
import { assetUrl, loadMemes } from './loader'
import { detectEvents, pickMeme, type MemeDefinition } from './triggers'

interface ActiveMeme extends MemeDefinition {
  /** Distinguishes repeats of the same meme so the animation restarts. */
  key: number
}

/**
 * Watches the game for anything worth reacting to and plays the matching
 * effect over the top of the screen.
 *
 * Rendered once, above everything, and entirely presentational — it never
 * touches game state.
 */
export function MemeOverlay() {
  const state = useMatch((s) => s.state)
  const [memes, setMemes] = useState<MemeDefinition[]>([])
  const [active, setActive] = useState<ActiveMeme | null>(null)
  const previous = useRef(state)
  const counter = useRef(0)

  useEffect(() => {
    const controller = new AbortController()
    void loadMemes(controller.signal).then(setMemes)
    return () => controller.abort()
  }, [])

  useEffect(() => {
    const before = previous.current
    previous.current = state
    if (!before || !state || memes.length === 0) return
    // A new game resets everything; nothing to celebrate.
    if (before.modeId !== state.modeId || state.history.length < before.history.length) return

    // One transition can raise several events at once — a checkout is also a
    // nine-darter and a game win. Pick the best meme across all of them rather
    // than taking the first event that happens to match something, or a
    // nine-darter gets announced as an ordinary good finish.
    let best: MemeDefinition | null = null
    for (const event of detectEvents(before, state)) {
      const meme = pickMeme(memes, event)
      if (meme && (best === null || (meme.priority ?? 0) > (best.priority ?? 0))) best = meme
    }

    if (best) {
      counter.current += 1
      setActive({ ...best, key: counter.current })
    }
  }, [state, memes])

  useEffect(() => {
    if (!active) return

    if (active.effect === 'confetti') {
      void confetti({
        particleCount: 120,
        spread: 80,
        origin: { y: 0.4 },
        colors: active.colour ? [active.colour, '#ffffff'] : undefined,
        disableForReducedMotion: true,
      })
    }

    if (active.sound) {
      const audio = new Audio(assetUrl(active.sound))
      audio.volume = 0.8
      // Autoplay can be refused before the first interaction; not worth
      // interrupting the game over.
      void audio.play().catch(() => undefined)
    }

    const timer = setTimeout(() => setActive(null), active.durationMs ?? 2000)
    return () => clearTimeout(timer)
  }, [active])

  if (!active) return null

  const colour = active.colour ?? '#fbbf24'

  return (
    <div
      key={active.key}
      data-testid={`meme-${active.id}`}
      className="pointer-events-none fixed inset-0 z-50 flex items-center justify-center overflow-hidden"
      aria-live="polite"
    >
      {/* Without a scrim the text competes with the keypad underneath and is
          genuinely hard to read at a glance across a room. */}
      <div className="meme-scrim absolute inset-0 bg-black/70 backdrop-blur-[2px]" />

      {active.effect === 'flash' && (
        <div className="meme-flash absolute inset-0" style={{ backgroundColor: colour }} />
      )}

      <div className={active.effect === 'shake' ? 'meme-shake' : 'meme-slam'}>
        {active.image && (
          <img
            src={assetUrl(active.image)}
            alt=""
            className="mx-auto max-h-[45vh] max-w-[80vw] object-contain drop-shadow-2xl"
          />
        )}
        {active.text && (
          <p
            className="px-6 text-center text-5xl font-black uppercase leading-none tracking-tight drop-shadow-[0_4px_12px_rgba(0,0,0,0.8)]"
            style={{ color: colour }}
          >
            {active.text}
          </p>
        )}
      </div>
    </div>
  )
}
