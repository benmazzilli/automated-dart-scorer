import { test, expect, type Page } from '@playwright/test'

type Ring = 'Double' | 'Treble'

async function throwDart(page: Page, number: number | 'MISS', ring?: Ring) {
  if (ring) await page.getByRole('button', { name: ring, exact: true }).click()
  await page.getByRole('button', { name: String(number), exact: true }).click()
}

async function throwVisit(page: Page, darts: [number | 'MISS', Ring?][]) {
  for (const [number, ring] of darts) await throwDart(page, number, ring)
}

const TON_EIGHTY: [number, Ring][] = [
  [20, 'Treble'],
  [20, 'Treble'],
  [20, 'Treble'],
]
const BLANK: ['MISS'][] = [['MISS'], ['MISS'], ['MISS']]

/** Play a full 501 leg where Ben wins in nine darts. */
async function playNineDarter(page: Page) {
  await page.goto('/')
  for (const name of ['Ben', 'Dave']) {
    await page.getByLabel('New player name').fill(name)
    await page.getByRole('button', { name: 'Add', exact: true }).click()
    await expect(page.getByTestId(`player-${name}`)).toBeVisible()
  }
  for (let i = 0; i < 21; i++) await page.getByRole('button', { name: 'Fewer legs' }).click()
  await page.getByRole('button', { name: 'Start game' }).click()

  for (let visit = 0; visit < 2; visit++) {
    await throwVisit(page, TON_EIGHTY)
    await throwVisit(page, BLANK)
  }
  // 141 left: T20, T15, D18.
  await throwVisit(page, [
    [20, 'Treble'],
    [15, 'Treble'],
    [18, 'Double'],
  ])
  await expect(page.getByRole('heading', { name: /Ben wins/ })).toBeVisible()
}

test.describe('profiles and stats', () => {
  test('remembers players between games', async ({ page }) => {
    await page.goto('/')
    await page.getByLabel('New player name').fill('Ben')
    await page.getByRole('button', { name: 'Add', exact: true }).click()
    await expect(page.getByTestId('player-Ben')).toBeVisible()

    await page.reload()
    await expect(page.getByTestId('player-Ben')).toBeVisible()
  })

  test('will not create the same player twice', async ({ page }) => {
    await page.goto('/')
    for (let i = 0; i < 2; i++) {
      await page.getByLabel('New player name').fill('Ben')
      await page.getByRole('button', { name: 'Add', exact: true }).click()
      await expect(page.getByTestId('player-Ben')).toBeVisible()
    }
    await expect(page.getByTestId('player-Ben')).toHaveCount(1)
  })

  test('records a finished match and computes statistics from it', async ({ page }) => {
    await playNineDarter(page)

    await page.getByRole('button', { name: 'New game' }).click()
    await page.getByRole('button', { name: 'Stats' }).click()

    await page.getByTestId('profile-Ben').click()

    // 501 in 9 darts is a 167 average, and the leg length is recorded.
    await expect(page.getByTestId('stat-3-dart average')).toContainText('167.0')
    await expect(page.getByTestId('stat-Best leg')).toContainText('9 darts')
    await expect(page.getByTestId('stat-Best finish')).toContainText('141')
    await expect(page.getByTestId('stat-Played')).toContainText('1')
    await expect(page.getByTestId('stat-Won')).toContainText('100%')
    await expect(page.getByText('vs Dave')).toBeVisible()
  })

  test('counts 180s and shows the head-to-head record', async ({ page }) => {
    await playNineDarter(page)
    await page.getByRole('button', { name: 'New game' }).click()
    await page.getByRole('button', { name: 'Stats' }).click()
    await page.getByTestId('profile-Ben').click()
    await expect(page.getByTestId('stat-180s')).toContainText('2')
    await expect(page.getByTestId('stat-Checkout')).toContainText('100%')

    // Dave lost, so his side of the record reads 0 won.
    await page.getByTestId('profile-Dave').click()
    await expect(page.getByText('vs Ben')).toBeVisible()
    await expect(page.getByTestId('stat-Won')).toContainText('0%')
  })

  test('shows nothing for a player who has not finished a game', async ({ page }) => {
    await page.goto('/')
    await page.getByLabel('New player name').fill('Ben')
    await page.getByRole('button', { name: 'Add', exact: true }).click()
    await expect(page.getByTestId('player-Ben')).toBeVisible()
    await page.getByRole('button', { name: 'Stats' }).click()
    await expect(page.getByText(/has not finished a game yet/)).toBeVisible()
  })
})
