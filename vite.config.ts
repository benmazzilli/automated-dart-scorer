import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { VitePWA } from 'vite-plugin-pwa'
import { fileURLToPath, URL } from 'node:url'

/**
 * Where the app is served from.
 *
 * Defaults to the site root, which is what dev, `vite preview` and the whole
 * Playwright suite run against. GitHub Pages serves a project site from
 * `/automated-dart-scorer/` instead, so the deploy workflow sets `VITE_BASE`.
 *
 * Kept as an override rather than hardcoded, because pinning the subpath here
 * would mean rewriting every `page.goto('/')` in the end-to-end tests —
 * Playwright resolves an absolute path against the origin, not against the
 * path component of `baseURL`.
 */
const base = process.env['VITE_BASE'] ?? '/'

export default defineConfig({
  base,
  plugins: [
    react(),
    tailwindcss(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['icon-192.png', 'icon-512.png'],
      manifest: {
        name: 'Oche — Darts Scorer',
        short_name: 'Oche',
        description: 'Camera dart scoring, classic and rogue games, local profiles and stats.',
        theme_color: '#0b0f14',
        background_color: '#0b0f14',
        display: 'standalone',
        orientation: 'portrait',
        // Both must follow the base, or an installed app launches at the domain
        // root and lands on a 404.
        scope: base,
        start_url: base,
        icons: [
          { src: 'icon-192.png', sizes: '192x192', type: 'image/png' },
          { src: 'icon-512.png', sizes: '512x512', type: 'image/png' },
          { src: 'icon-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
        ],
      },
      workbox: {
        // Meme assets are user-supplied at runtime, so never precache them.
        globPatterns: ['**/*.{js,css,html,png,svg,woff2}'],
        globIgnores: ['memes/**'],
      },
    }),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  test: {
    globals: true,
    environment: 'node',
    include: ['tests/**/*.test.ts', 'src/**/*.test.ts'],
  },
})
