import { resolve } from 'node:path'
import type { DrawDartOptions } from '../../src/vision/testing'

/** Where the generated fake camera feed is written. Gitignored. */
export const FAKE_VIDEO_PATH = resolve('test-results/fake-camera/board.y4m')

/** Calibration taps for the generated scene, written alongside the video. */
export const FAKE_VIDEO_TAPS = resolve('test-results/fake-camera/taps.json')

/**
 * The darts that land in the fake feed, in order.
 *
 * Chosen to be unambiguous: the middle of a bed rather than near a wire, and
 * one of each ring so single, treble and double are all exercised.
 */
export const FAKE_VIDEO_DARTS: DrawDartOptions[] = [
  { radiusMm: 103, bearingDeg: 0 }, // T20
  { radiusMm: 130, bearingDeg: 90 }, // 6
  { radiusMm: 166, bearingDeg: 180 }, // D3
]

/** What those darts should score, in order. */
export const FAKE_VIDEO_EXPECTED = ['T20', '6', 'D3'] as const
