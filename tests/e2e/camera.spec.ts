import { readFileSync } from 'node:fs'
import { test, expect, type Page } from '@playwright/test'
import { FAKE_VIDEO_EXPECTED, FAKE_VIDEO_TAPS } from './fake-video'

interface TapsFile {
  width: number
  height: number
  taps: { x: number; y: number }[]
}

const { width, height, taps } = JSON.parse(readFileSync(FAKE_VIDEO_TAPS, 'utf8')) as TapsFile

async function startGame(page: Page) {
  await page.goto('/')
  for (const name of ['Ben', 'Dave']) {
    await page.getByLabel('New player name').fill(name)
    await page.getByRole('button', { name: 'Add', exact: true }).click()
    await expect(page.getByTestId(`player-${name}`)).toBeVisible()
  }
  await page.getByRole('button', { name: 'Start game' }).click()
  // Keypad → Board → Camera.
  await page.getByRole('button', { name: 'Board', exact: true }).click()
  await page.getByRole('button', { name: 'Camera', exact: true }).click()
}

/** Tap the four calibration points on the preview, in order. */
async function calibrate(page: Page) {
  const preview = page.locator('video').locator('..')
  await expect(preview).toBeVisible()
  await expect(page.getByText('Top of the 20')).toBeVisible({ timeout: 20_000 })

  const box = (await preview.boundingBox())!
  for (const tap of taps) {
    await page.mouse.click(
      box.x + (tap.x / width) * box.width,
      box.y + (tap.y / height) * box.height,
    )
  }
}

test.describe('camera scoring', () => {
  test('starts the camera and asks to be calibrated', async ({ page }) => {
    await startGame(page)
    await expect(page.getByText('Calibrate · 1 of 4')).toBeVisible({ timeout: 20_000 })
    await expect(page.getByText('Top of the 20')).toBeVisible()
  })

  test('steps through the four calibration targets', async ({ page }) => {
    await startGame(page)
    const preview = page.locator('video').locator('..')
    await expect(page.getByText('Top of the 20')).toBeVisible({ timeout: 20_000 })

    const box = (await preview.boundingBox())!
    await page.mouse.click(
      box.x + (taps[0]!.x / width) * box.width,
      box.y + (taps[0]!.y / height) * box.height,
    )
    await expect(page.getByText('Calibrate · 2 of 4')).toBeVisible()
    await expect(page.getByText('Right of the 6')).toBeVisible()
  })

  test('rejects taps that do not describe a board', async ({ page }) => {
    await startGame(page)
    await expect(page.getByText('Top of the 20')).toBeVisible({ timeout: 20_000 })

    page.on('dialog', (dialog) => void dialog.accept())
    const preview = page.locator('video').locator('..')
    const box = (await preview.boundingBox())!
    // Four taps in a tight cluster: solvable, but nothing like a board.
    for (let i = 0; i < 4; i++) {
      await page.mouse.click(box.x + box.width / 2 + i * 3, box.y + box.height / 2 + i * 3)
    }
    // Calibration is refused, so it starts over from the first target.
    await expect(page.getByText('Calibrate · 1 of 4')).toBeVisible()
  })

  test('calibrates and then scores the darts as they land', async ({ page }) => {
    // The feed runs a five-second cycle and each reading pauses before it is
    // accepted, so a full visit needs longer than the default budget.
    test.setTimeout(120_000)
    await startGame(page)
    await calibrate(page)

    // Calibration done: the prompt is replaced by the watching indicator.
    await expect(page.getByText('Calibrate · 1 of 4')).toBeHidden()
    await expect(page.getByRole('button', { name: 'Recalibrate' })).toBeVisible()

    // The feed drops three darts in on a loop, auto-accepted when confident,
    // so the visit fills in without any further tapping.
    //
    // Only the first two darts can be read from the panel: applying the third
    // completes the visit, which banks it and clears all three slots in the
    // same render, so slot three is never painted. The third dart is verified
    // through the score instead.
    const thrown: (string | null)[] = [null, null]
    const deadline = Date.now() + 60_000
    while (Date.now() < deadline && thrown.some((value) => value === null)) {
      for (let index = 0; index < thrown.length; index++) {
        if (thrown[index] !== null) continue
        const text = (await page.getByTestId(`dart-${index}`).textContent())?.trim()
        if (text && text !== '–') thrown[index] = text
      }
      await page.waitForTimeout(100)
    }

    // Calibration finishes at an unpredictable point in the looping feed, so
    // the visit can begin on any of the three darts. Whichever it starts on,
    // the pair must be consecutive in the feed's order.
    const expected = [...FAKE_VIDEO_EXPECTED]
    const consecutivePairs = expected.map((dart, i) => [dart, expected[(i + 1) % expected.length]!])
    expect(consecutivePairs, `saw ${JSON.stringify(thrown)}`).toContainEqual(thrown)

    // All three darts total 72 whatever the rotation, so the score settling on
    // 501 - 72 = 429 is what confirms the third one was read correctly too.
    await expect(page.getByTestId('score-Ben')).toHaveText('429', { timeout: 30_000 })
  })

  test('keeps the calibration across a reload', async ({ page }) => {
    await startGame(page)
    await calibrate(page)
    await expect(page.getByRole('button', { name: 'Recalibrate' })).toBeVisible()

    // The match itself is not persisted — only the calibration is, since a
    // phone left on its stand keeps the same view of the board between games.
    await page.reload()
    await startGame(page)
    await expect(page.getByRole('button', { name: 'Recalibrate' })).toBeVisible({ timeout: 20_000 })
    await expect(page.getByText('Calibrate · 1 of 4')).toBeHidden()
  })

  test('can be recalibrated', async ({ page }) => {
    await startGame(page)
    await calibrate(page)
    await page.getByRole('button', { name: 'Recalibrate' }).click()
    await expect(page.getByText('Calibrate · 1 of 4')).toBeVisible()
  })

  test('offers a correction path for a wrong reading', async ({ page }) => {
    await startGame(page)
    await calibrate(page)

    // Wait for a reading, then reject it and place the dart by hand.
    await expect(page.getByRole('button', { name: 'Wrong' })).toBeVisible({ timeout: 30_000 })
    await page.getByRole('button', { name: 'Wrong' }).click()
    await expect(page.getByText('Tap where the dart actually landed')).toBeVisible()

    const board = page.getByRole('button', { name: /Dartboard/ })
    const box = (await board.boundingBox())!
    await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2)
    await expect(page.getByTestId('dart-0')).toHaveText('BULL')
  })
})
