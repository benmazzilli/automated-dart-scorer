import { useState } from 'react'

const DISMISSED_KEY = 'oche-guide-dismissed'

/**
 * First-run notes on getting the camera to work.
 *
 * Worth showing once, because the two things that decide whether auto-scoring
 * is any good are both about how the phone is placed, and neither is obvious.
 */
export function SetupGuide() {
  const [dismissed, setDismissed] = useState(() => {
    try {
      return localStorage.getItem(DISMISSED_KEY) === '1'
    } catch {
      return false
    }
  })

  if (dismissed) return null

  function dismiss() {
    try {
      localStorage.setItem(DISMISSED_KEY, '1')
    } catch {
      // Private browsing refuses writes; the panel just returns next launch.
    }
    setDismissed(true)
  }

  return (
    <section className="rounded-xl border border-neutral-800 bg-neutral-900/60 p-4 text-sm">
      <h2 className="font-bold text-amber-400">Setting up the camera</h2>
      <ul className="mt-2 flex list-disc flex-col gap-1.5 pl-4 text-neutral-300">
        <li>
          Stand the phone <strong>square on to the board</strong>, at about board height, a metre
          or two back. Off to one side and the scoring drifts.
        </li>
        <li>
          Keep it <strong>still</strong>. Detection works by spotting what changed, so a phone that
          gets nudged has to be recalibrated.
        </li>
        <li>Even, steady light. A hard shadow moving across the board reads as movement.</li>
        <li>
          Calibrate by tapping the outer edge of the double ring at the 20, the 6, the 3 and the
          11, in that order.
        </li>
        <li>
          One camera cannot judge depth, so overlapping darts are the weak spot. Every reading can
          be corrected with a tap, and the keypad is always there.
        </li>
      </ul>
      <button
        type="button"
        onClick={dismiss}
        className="mt-3 rounded-lg bg-neutral-800 px-4 py-2 text-xs font-semibold text-neutral-200"
      >
        Got it
      </button>
    </section>
  )
}
