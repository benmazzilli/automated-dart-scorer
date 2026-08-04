import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname } from 'node:path'
import { DEFAULT_VISION_CONFIG } from '../../src/vision/config'
import type { GrayImage } from '../../src/vision/frameDiff'
import {
  drawHand,
  frameWithDarts,
  makeBackground,
  makeScene,
  type DrawDartOptions,
} from '../../src/vision/testing'
import { FAKE_VIDEO_PATH, FAKE_VIDEO_TAPS, FAKE_VIDEO_DARTS } from './fake-video'

/**
 * Build a Y4M video of a synthetic board with darts arriving one at a time,
 * and hand out the tap coordinates that calibrate it.
 *
 * Chromium can be pointed at a file in place of a real camera, so this drives
 * the actual `getUserMedia` path, the actual frame loop and the actual
 * detection — everything the phone will run except the phone's optics.
 */
const { frameWidth: WIDTH, frameHeight: HEIGHT, stableFrames } = DEFAULT_VISION_CONFIG

/** Enough still frames to clear the settle threshold with room to spare. */
const SETTLE_FRAMES = stableFrames + 5
const MOTION_FRAMES = 4
const LEAD_IN_FRAMES = 8

function y4mHeader(): Buffer {
  return Buffer.from(`YUV4MPEG2 W${WIDTH} H${HEIGHT} F10:1 Ip A1:1 C420\n`, 'ascii')
}

/** One greyscale frame as YUV 4:2:0, with the colour planes left neutral. */
function y4mFrame(image: GrayImage): Buffer {
  const luma = Buffer.from(image.data)
  const chromaSize = (WIDTH / 2) * (HEIGHT / 2)
  const chroma = Buffer.alloc(chromaSize * 2, 128)
  return Buffer.concat([Buffer.from('FRAME\n', 'ascii'), luma, chroma])
}

function repeat(image: GrayImage, times: number): Buffer[] {
  return Array.from({ length: times }, () => y4mFrame(image))
}

export default function globalSetup(): void {
  const scene = makeScene(WIDTH, HEIGHT)
  const background = makeBackground(scene)
  const hand = drawHand(scene, background)

  const chunks: Buffer[] = [y4mHeader()]

  // Empty board first, so the watcher has a baseline to compare against.
  chunks.push(...repeat(background, LEAD_IN_FRAMES))

  // Then each dart in turn: movement, then the board at rest with one more
  // dart in it than before.
  const landed: DrawDartOptions[] = []
  for (const dart of FAKE_VIDEO_DARTS) {
    landed.push(dart)
    chunks.push(...repeat(hand, MOTION_FRAMES))
    chunks.push(...repeat(frameWithDarts(scene, background, landed), SETTLE_FRAMES))
  }

  mkdirSync(dirname(FAKE_VIDEO_PATH), { recursive: true })
  writeFileSync(FAKE_VIDEO_PATH, Buffer.concat(chunks))

  // The calibration taps are a property of the generated scene, so they are
  // written out beside it rather than hardcoded in the test.
  writeFileSync(
    FAKE_VIDEO_TAPS,
    JSON.stringify({ width: WIDTH, height: HEIGHT, taps: scene.calibration.taps }),
  )
}
