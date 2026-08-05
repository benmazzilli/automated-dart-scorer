import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { calibrate, type CalibrationPoints } from '../vision/calibration'
import { DEFAULT_VISION_CONFIG, type VisionConfig } from '../vision/config'

interface VisionStore {
  /** The four taps, if the board has been calibrated this session. */
  taps: CalibrationPoints | null
  config: VisionConfig
  /** Whether to draw the difference mask and detected tip over the preview. */
  debug: boolean

  setTaps(taps: CalibrationPoints | null): void
  setConfig(patch: Partial<VisionConfig>): void
  resetConfig(): void
  toggleDebug(): void
}

/**
 * Calibration and vision settings.
 *
 * Persisted because a phone left on its stand keeps the same view between
 * games, so re-tapping every time would be tedious. Only the taps are stored —
 * the homography is derived, and keeping the derived value would risk it
 * drifting out of step with the points it came from.
 */
export const useVision = create<VisionStore>()(
  persist(
    (set) => ({
      taps: null,
      config: DEFAULT_VISION_CONFIG,
      debug: false,

      setTaps(taps) {
        set({ taps })
      },
      setConfig(patch) {
        set((state) => ({ config: { ...state.config, ...patch } }))
      },
      resetConfig() {
        set({ config: DEFAULT_VISION_CONFIG })
      },
      toggleDebug() {
        set((state) => ({ debug: !state.debug }))
      },
    }),
    {
      name: 'oche-vision',
      // Config gains fields over time; merge so a stored older shape does not
      // leave new thresholds undefined.
      merge: (persisted, current) => {
        const saved = persisted as Partial<VisionStore> | undefined
        return {
          ...current,
          ...saved,
          config: { ...DEFAULT_VISION_CONFIG, ...(saved?.config ?? {}) },
        }
      },
    },
  ),
)

/** The calibration derived from the stored taps, or `null` if not calibrated. */
export function useCalibration() {
  const taps = useVision((s) => s.taps)
  return taps ? calibrate(taps) : null
}
