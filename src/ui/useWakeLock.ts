import { useEffect } from 'react'

/**
 * Hold the screen awake while a game is in progress.
 *
 * A phone on a stand watching the board gets no touch input for minutes at a
 * time, so it dims and locks in the middle of a leg — which also stops the
 * camera. Supported from iOS 16.4; older versions simply carry on without it.
 */
export function useWakeLock(active: boolean): void {
  useEffect(() => {
    if (!active || !('wakeLock' in navigator)) return

    let sentinel: WakeLockSentinel | null = null
    let released = false

    async function acquire() {
      try {
        sentinel = await navigator.wakeLock.request('screen')
      } catch {
        // Refused when the tab is hidden or the battery is low. Not worth
        // surfacing — the game plays fine, the screen just sleeps.
      }
    }

    // The lock is dropped whenever the tab is backgrounded, so it has to be
    // taken again on return rather than assumed to still be held.
    function handleVisibility() {
      if (document.visibilityState === 'visible' && !released) void acquire()
    }

    void acquire()
    document.addEventListener('visibilitychange', handleVisibility)

    return () => {
      released = true
      document.removeEventListener('visibilitychange', handleVisibility)
      void sentinel?.release().catch(() => undefined)
    }
  }, [active])
}
