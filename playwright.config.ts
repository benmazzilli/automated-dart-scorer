import { existsSync } from 'node:fs'
import { defineConfig, devices } from '@playwright/test'

/**
 * The development container ships Chromium at a fixed path with
 * PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD set, so tests use that rather than
 * downloading a browser. On CI the path does not exist and Playwright's own
 * managed browser is used instead.
 */
const containerChromium = process.env['CHROMIUM_PATH'] ?? '/opt/pw-browsers/chromium'
const executablePath = existsSync(containerChromium) ? containerChromium : undefined

import { FAKE_VIDEO_PATH } from './tests/e2e/fake-video'

/**
 * Chromium can take a Y4M file in place of a camera, which lets the camera
 * scoring path be driven end to end — real getUserMedia, real frame loop, real
 * detection — without a board or a phone.
 */
const FAKE_CAMERA_ARGS = [
  '--use-fake-device-for-media-stream',
  '--use-fake-ui-for-media-stream',
  `--use-file-for-fake-video-capture=${FAKE_VIDEO_PATH}`,
]

export default defineConfig({
  testDir: './tests/e2e',
  globalSetup: './tests/e2e/global-setup.ts',
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
      // The camera suite needs the fake-device flags, so it runs in its own
      // project rather than here.
      testIgnore: /camera.*\.spec\.ts/,
      use: {
        ...devices['iPhone 13'],
        // iPhone 13 defaults to WebKit; run the engine we actually have.
        browserName: 'chromium',
        defaultBrowserType: 'chromium',
        launchOptions: executablePath ? { executablePath } : {},
      },
    },
    {
      name: 'fake-camera',
      testMatch: /camera.*\.spec\.ts/,
      use: {
        ...devices['iPhone 13'],
        browserName: 'chromium',
        defaultBrowserType: 'chromium',
        permissions: ['camera'],
        launchOptions: executablePath
          ? { executablePath, args: FAKE_CAMERA_ARGS }
          : { args: FAKE_CAMERA_ARGS },
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
