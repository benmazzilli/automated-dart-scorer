import { test, expect } from '@playwright/test'

/**
 * Not an assertion — this renders each screen to an image so the layout can be
 * looked at rather than guessed about. Run with `npx playwright test screenshot`.
 */
test('capture screens', async ({ page }) => {
  await page.goto('/')
  for (const name of ['Ben', 'Dave']) {
    await page.getByLabel('New player name').fill(name)
    await page.getByRole('button', { name: 'Add', exact: true }).click()
    await expect(page.getByTestId(`player-${name}`)).toBeVisible()
  }
  await page.screenshot({ path: 'test-results/shots/1-setup.png', fullPage: true })

  await page.getByRole('button', { name: 'Start game' }).click()
  await page.screenshot({ path: 'test-results/shots/2-play-keypad.png', fullPage: true })

  await page.getByRole('button', { name: 'Treble', exact: true }).click()
  await page.getByRole('button', { name: '20', exact: true }).click()
  await page.getByRole('button', { name: 'Treble', exact: true }).click()
  await page.getByRole('button', { name: '20', exact: true }).click()
  await page.screenshot({ path: 'test-results/shots/3-play-scored.png', fullPage: true })

  await page.getByRole('button', { name: 'Board', exact: true }).click()
  await page.screenshot({ path: 'test-results/shots/4-play-board.png', fullPage: true })
})
