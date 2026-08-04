import { test, expect, type Page } from '@playwright/test'

type Ring = 'Double' | 'Treble'

/** Enter one dart on the keypad. */
async function throwDart(page: Page, number: number | 'BULL' | '25' | 'MISS', ring?: Ring) {
  if (ring) await page.getByRole('button', { name: ring, exact: true }).click()
  await page.getByRole('button', { name: String(number), exact: true }).click()
}

async function throwVisit(page: Page, darts: [number | 'MISS', Ring?][]) {
  for (const [number, ring] of darts) await throwDart(page, number, ring)
}

const MAX_LEGS = 21

async function startGame(page: Page, options: { mode?: string; legs?: number } = {}) {
  await page.goto('/')
  if (options.mode) await page.getByRole('button', { name: new RegExp(options.mode) }).click()
  if (options.legs !== undefined) {
    for (let i = 0; i < MAX_LEGS; i++) await page.getByRole('button', { name: 'Fewer legs' }).click()
    for (let i = 1; i < options.legs; i++) await page.getByRole('button', { name: 'More legs' }).click()
  }
  await page.getByRole('button', { name: 'Start game' }).click()
}

const score = (page: Page, player: string) => page.getByTestId(`score-${player}`)

test.describe('501', () => {
  test('scores a turn and hands over to the next player', async ({ page }) => {
    await startGame(page)

    await expect(score(page, 'Player 1')).toHaveText('501')
    await throwVisit(page, [
      [20, 'Treble'],
      [20, 'Treble'],
      [20, 'Treble'],
    ])

    await expect(score(page, 'Player 1')).toHaveText('321')
    // The turn is banked and the darts panel is cleared for Player 2.
    await expect(page.getByTestId('dart-0')).toHaveText('–')
  })

  test('resets the multiplier after each dart so trebles do not stick', async ({ page }) => {
    await startGame(page)
    await throwDart(page, 20, 'Treble')
    await throwDart(page, 20) // a plain 20, not another treble
    await expect(page.getByTestId('dart-0')).toHaveText('T20')
    await expect(page.getByTestId('dart-1')).toHaveText('20')
    await expect(score(page, 'Player 1')).toHaveText('421')
  })

  test('shows a checkout suggestion once one is available', async ({ page }) => {
    await startGame(page)
    // 501 has no finish, so nothing is suggested yet.
    await expect(page.getByText(/T20 D20/)).toBeHidden()

    // Walk Player 1 to exactly 100 at the start of a visit, so all three darts
    // are available and the suggested route is the three-dart one.
    // 501 - 180 - 180 - 41 = 100.
    const blank: ['MISS'][] = [['MISS'], ['MISS'], ['MISS']]
    for (let i = 0; i < 2; i++) {
      await throwVisit(page, [
        [20, 'Treble'],
        [20, 'Treble'],
        [20, 'Treble'],
      ])
      await throwVisit(page, blank)
    }
    await expect(score(page, 'Player 1')).toHaveText('141')
    await throwVisit(page, [[20], [20], [1]]) // 41
    await throwVisit(page, blank)

    await expect(score(page, 'Player 1')).toHaveText('100')
    await expect(page.getByText('100 — T20 D20')).toBeVisible()
  })

  test('busts the whole turn and restores the score', async ({ page }) => {
    await startGame(page)
    // Walk Player 1 down to 40: 501 - 180 - 180 - 101 = 40.
    await throwVisit(page, [
      [20, 'Treble'],
      [20, 'Treble'],
      [20, 'Treble'],
    ])
    await throwVisit(page, [['MISS'], ['MISS'], ['MISS']])
    await throwVisit(page, [
      [20, 'Treble'],
      [20, 'Treble'],
      [20, 'Treble'],
    ])
    await throwVisit(page, [['MISS'], ['MISS'], ['MISS']])
    await throwVisit(page, [
      [20, 'Treble'],
      [20, 'Treble'],
      [20, 'Double'],
    ]) // 141 -> 40 exactly? 501-360 = 141; 60+60+40 = 160 -> bust
    // That visit busts by going below zero, so the score stays at 141.
    await expect(page.getByText(/BUST/)).toBeVisible()
    await expect(score(page, 'Player 1')).toHaveText('141')
  })

  test('undoes a dart and a whole turn', async ({ page }) => {
    await startGame(page)
    await throwDart(page, 20, 'Treble')
    await expect(score(page, 'Player 1')).toHaveText('441')

    await page.getByRole('button', { name: 'Undo dart' }).click()
    await expect(score(page, 'Player 1')).toHaveText('501')

    // Bank a full turn, then take the whole thing back.
    await throwVisit(page, [
      [20, 'Treble'],
      [20, 'Treble'],
      [20, 'Treble'],
    ])
    await expect(score(page, 'Player 1')).toHaveText('321')
    await page.getByRole('button', { name: 'Undo turn' }).click()
    await expect(score(page, 'Player 1')).toHaveText('501')
    await expect(page.getByTestId('dart-0')).toHaveText('–')
  })

  test('plays a leg through to the win screen', async ({ page }) => {
    await startGame(page, { legs: 1 })

    // 501 = 180 + 180 + 141, with Player 2 blanking in between.
    for (let visit = 0; visit < 2; visit++) {
      await throwVisit(page, [
        [20, 'Treble'],
        [20, 'Treble'],
        [20, 'Treble'],
      ])
      await throwVisit(page, [['MISS'], ['MISS'], ['MISS']])
    }

    await expect(score(page, 'Player 1')).toHaveText('141')
    await throwVisit(page, [
      [20, 'Treble'],
      [15, 'Treble'],
      [18, 'Double'],
    ])

    await expect(page.getByRole('heading', { name: /Player 1 wins/ })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Play again' })).toBeVisible()
  })
})

test.describe('board entry', () => {
  test('scores the bull by tapping the centre', async ({ page }) => {
    await startGame(page)
    await page.getByRole('button', { name: 'Board', exact: true }).click()

    const board = page.getByRole('button', { name: /Dartboard/ })
    await expect(board).toBeVisible()

    const box = (await board.boundingBox())!
    await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2)

    await expect(page.getByTestId('dart-0')).toHaveText('BULL')
    await expect(score(page, 'Player 1')).toHaveText('451')
  })

  test('scores the 20 bed by tapping straight up from the centre', async ({ page }) => {
    await startGame(page)
    await page.getByRole('button', { name: 'Board', exact: true }).click()

    const box = (await page.getByRole('button', { name: /Dartboard/ }).boundingBox())!
    const centreX = box.x + box.width / 2
    const centreY = box.y + box.height / 2
    // 40% of the way to the edge, straight up: the large single 20.
    await page.mouse.click(centreX, centreY - box.height * 0.2)

    await expect(page.getByTestId('dart-0')).toHaveText('20')
  })
})

test.describe('other modes', () => {
  test('Around the Clock tracks the target', async ({ page }) => {
    await startGame(page, { mode: 'Around the Clock' })
    await expect(page.getByText('Target: 1')).toBeVisible()
    await throwDart(page, 1)
    await expect(page.getByText('Target: 2')).toBeVisible()
  })

  test('Killer needs a double to arm', async ({ page }) => {
    await startGame(page, { mode: 'Killer' })
    await expect(page.getByText(/Hit D\d+ to become a killer/)).toBeVisible()
  })

  test('Halve It names the round target', async ({ page }) => {
    await startGame(page, { mode: 'Halve It' })
    await expect(page.getByText('Target: 20')).toBeVisible()
  })

  test('Shanghai names the round target', async ({ page }) => {
    await startGame(page, { mode: 'Shanghai' })
    await expect(page.getByText(/Round 1 of 7 — target 1/)).toBeVisible()
  })
})
