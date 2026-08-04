import { defineConfig, devices } from '@playwright/test'

/**
 * The container ships Chromium at a fixed path with
 * PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD set, so tests use that rather than
 * downloading a browser.
 */
const executablePath = process.env['CHROMIUM_PATH'] ?? '/opt/pw-browsers/chromium'

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  forbidOnly: !!process.env['CI'],
  retries: process.env['CI'] ? 2 : 0,
  reporter: process.env['CI'] ? 'list' : [['list']],
  use: {
    baseURL: 'http://127.0.0.1:4173',
    trace: 'on-first-retry',
  },
  projects: [
    {
      name: 'mobile-safari-sized',
      use: {
        ...devices['iPhone 13'],
        // iPhone 13 defaults to WebKit; run the engine we actually have.
        browserName: 'chromium',
        defaultBrowserType: 'chromium',
        launchOptions: { executablePath },
      },
    },
  ],
  webServer: {
    command: 'npm run preview -- --port 4173 --host 127.0.0.1',
    url: 'http://127.0.0.1:4173',
    reuseExistingServer: !process.env['CI'],
    timeout: 120_000,
  },
})
