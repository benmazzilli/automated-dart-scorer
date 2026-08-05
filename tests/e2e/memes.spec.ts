import { test, expect, type Page } from '@playwright/test'

type Ring = 'Double' | 'Treble'

async function throwDart(page: Page, number: number | 'MISS', ring?: Ring) {
  if (ring) await page.getByRole('button', { name: ring, exact: true }).click()
  await page.getByRole('button', { name: String(number), exact: true }).click()
}

async function startGame(page: Page, legs = 3) {
  await page.goto('/')
  for (const name of ['Ben', 'Dave']) {
    await page.getByLabel('New player name').fill(name)
    await page.getByRole('button', { name: 'Add', exact: true }).click()
    await expect(page.getByTestId(`player-${name}`)).toBeVisible()
  }
  for (let i = 0; i < 21; i++) await page.getByRole('button', { name: 'Fewer legs' }).click()
  for (let i = 1; i < legs; i++) await page.getByRole('button', { name: 'More legs' }).click()
  await page.getByRole('button', { name: 'Start game' }).click()
}

test.describe('meme effects', () => {
  test('celebrates a 180', async ({ page }) => {
    await startGame(page)
    await throwDart(page, 20, 'Treble')
    await throwDart(page, 20, 'Treble')
    await throwDart(page, 20, 'Treble')

    const meme = page.getByTestId('meme-one-eighty')
    await expect(meme).toBeVisible()
    await expect(meme).toContainText('ONE HUNDRED AND EIGHTY')
  })

  test('picks the 180 over the lesser scoring memes', async ({ page }) => {
    await startGame(page)
    await throwDart(page, 20, 'Treble')
    await throwDart(page, 20, 'Treble')
    await throwDart(page, 20, 'Treble')
    // 180 also satisfies the 100+ and 140+ rules; only one effect shows.
    await expect(page.getByTestId('meme-ton')).toBeHidden()
    await expect(page.getByTestId('meme-ton-forty')).toBeHidden()
  })

  test('reacts to a bag o’ nuts', async ({ page }) => {
    await startGame(page)
    // 26 is the classic bad visit: 20, 5, 1.
    await throwDart(page, 20)
    await throwDart(page, 5)
    await throwDart(page, 1)
    await expect(page.getByTestId('meme-bag-o-nuts')).toBeVisible()
  })

  test('reacts to a bust', async ({ page }) => {
    await startGame(page, 1)
    // Get to 40, then bust it with a treble.
    for (let visit = 0; visit < 2; visit++) {
      for (let i = 0; i < 3; i++) await throwDart(page, 20, 'Treble')
      for (let i = 0; i < 3; i++) await throwDart(page, 'MISS')
    }
    // 141 left; T20 T20 D20 goes below zero.
    await throwDart(page, 20, 'Treble')
    await throwDart(page, 20, 'Treble')
    await throwDart(page, 20, 'Double')
    await expect(page.getByTestId('meme-bust')).toBeVisible()
  })

  test('goes off for a big finish', async ({ page }) => {
    await startGame(page, 1)
    for (let visit = 0; visit < 2; visit++) {
      for (let i = 0; i < 3; i++) await throwDart(page, 20, 'Treble')
      for (let i = 0; i < 3; i++) await throwDart(page, 'MISS')
    }
    // 141 checkout: T20 T15 D18. Also a nine-darter, which outranks it.
    await throwDart(page, 20, 'Treble')
    await throwDart(page, 15, 'Treble')
    await throwDart(page, 18, 'Double')
    await expect(page.getByTestId('meme-nine-darter')).toBeVisible()
  })

  test('says nothing about an ordinary visit', async ({ page }) => {
    await startGame(page)
    await throwDart(page, 5)
    await throwDart(page, 20)
    await throwDart(page, 20)
    // 45 is unremarkable, so no overlay at all.
    await expect(page.locator('[data-testid^="meme-"]')).toHaveCount(0)
  })
})
