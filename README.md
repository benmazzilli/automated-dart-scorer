# Oche — darts scorer

A mini darts app for the phone. Prop it on a stand facing the board and it
scores your throws from the camera; when it gets one wrong, fix it with a tap.
Five game modes, player profiles with stats that stick around, and effects that
go off when someone hits a 180 or throws 26.

Everything runs on the device. No account, no server, no signal needed.

---

## Getting started

```bash
npm install
npm run dev
```

Open the address it prints. On an iPhone, load it in Safari over HTTPS and use
**Share → Add to Home Screen** to install it.

| Command | Does |
| --- | --- |
| `npm run dev` | Development server |
| `npm run build` | Production build into `dist/` |
| `npm test` | Unit tests |
| `npm run typecheck` | TypeScript, no emit |
| `npm run e2e` | Playwright tests, including the camera path |

## Playing

Add everyone throwing, pick a game, and start. Three ways to enter a score,
switched with the button under the board:

- **Keypad** — pick single/double/treble, then the number.
- **Board** — tap where the dart landed.
- **Camera** — calibrate once, then it scores by itself.

Undo works one dart at a time or a whole visit at a time, which matters when the
camera misreads.

### Games

| Game | Rules |
| --- | --- |
| **501 / 301** | Race to zero. Straight or double in and out, legs and sets, checkout suggestions. |
| **Killer** | Hit your own double to become a killer, then take everyone else's lives. |
| **Around the Clock** | 1 to 20 in order, then the bull. |
| **Shanghai** | Round *n* is played on the *n*. Single, double and treble of it wins outright. |
| **Halve It** | Hit the round's target or lose half your score. |

### Stats

Kept per player, across every game: three-dart and first-nine averages, checkout
rate, best leg, highest finish, counts of 180s, 140s and tons, busts, and a
head-to-head record against everyone you have played.

Matches are stored as their full list of visits rather than as summed-up
figures, so a statistic added later applies to games you have already played.

## The camera

### Setting up

Stand the phone **square on to the board**, roughly at board height, a metre or
two back, and **keep it still**. Then calibrate by tapping the outer edge of the
double ring at the **20**, the **6**, the **3** and the **11**, in that order.
The calibration is remembered, so a phone left on its stand does not need
re-tapping between games.

### How it works

Rather than trying to find "a dart" in a picture — which is hard — it watches
for *what changed*. The scene is still, something moves, it goes still again,
and whatever is different now is the dart that just landed.

1. Frames are sampled at 10fps down to 640×480.
2. Each frame is compared against the last settled one. Lots of change means an
   arm or a dart in flight, so nothing is read.
3. Once it has been still for half a second, the difference is thresholded,
   cleaned up with morphological open and close, and split into connected
   components.
4. The largest component is checked for size and elongation — a dart is a long
   thin thing, a hand is round and gets rejected.
5. Its long axis comes from the covariance eigenvector, and the end nearer the
   bull is the tip. The flight is closer to the lens, so it projects further
   out.
6. The tip goes through the calibration homography into board millimetres, and
   from there into a score.

All of it is written out longhand in TypeScript. OpenCV.js would be an 8–10MB
WebAssembly download, which is not a sensible thing to make a phone fetch at the
oche.

### What it will and will not do

A single camera cannot judge depth. A dart pointing at the lens hides where it
actually went in, and no amount of care in the code recovers that — the
commercial systems use three cameras and triangulate. For reference,
[DeepDarts](https://arxiv.org/abs/2105.09880), the published single-camera work,
scored 94.7% correct on its own board and noticeably worse on a different one.

So: expect it to be good, not perfect. Overlapping darts and bounce-outs are the
weak spots. Every reading can be corrected with a tap, low-confidence readings
wait to be confirmed rather than counting themselves, and the keypad is always
there.

**The thresholds have been tuned against generated frames, not a real board.**
They live in one file, `src/vision/config.ts`, and the Debug button overlays the
live difference mask on the preview so you can see what it is reacting to.
Expect to adjust them once you point it at your own board and lighting.

## Memes

Effects fire on 180, 140+, tons, 26, busts, a dreadful visit, checkouts, big
finishes, nine-darters and whitewashes. The built-in ones are original — text,
shake, confetti, colour flash — and no third-party assets ship with the repo.

To add your own, drop images or sounds in `public/memes/` and map them to
triggers in `memes.config.json`. It is fetched at runtime, so nothing needs
rebuilding. See [`public/memes/README.md`](public/memes/README.md).

## How it is built

React, TypeScript and Vite, with Zustand for state, Dexie over IndexedDB for
storage, and Tailwind for styling. Installable as a PWA through
`vite-plugin-pwa`.

```
src/
  game/      board geometry, rules engine, modes, checkouts, stats
  vision/    homography, frame differencing, blobs, tip finding, camera
  memes/     triggers, effects, runtime asset loading
  db/        Dexie schema and queries
  ui/        screens and components
  store/     match and vision state
```

Two pieces are worth knowing about:

**`src/game/board.ts` is the only place board geometry lives.** Everything that
turns a position into a score goes through it. The previous version of this
project kept geometry in two places that disagreed, so scores were only right
when the board happened to be photographed at one particular orientation.

**Game modes are pure reducers.** `applyThrow(state, dart)` returns a new state
and never mutates, which is what makes undo and replay fall out for free — and
that is not a luxury when a camera is doing the scoring.

## Tests

199 unit tests and 31 end-to-end tests.

The end-to-end suite drives real Chromium at an iPhone viewport. The camera
tests are the interesting ones: a Y4M video of a synthetic board with darts
landing one by one is generated at test time and handed to Chromium in place of
a camera, so `getUserMedia`, the frame loop and the detector all genuinely run.
The suite calibrates through the UI and checks that a full visit scores T20, 6
and D3 to leave 429.

That verifies the geometry and the logic. It cannot verify the thresholds hold
up under real light — only your board can do that.

## Deploying

The build output is static, so anything that serves files will do. Camera access
needs HTTPS.

```bash
npm run build     # -> dist/
```

Cloudflare Pages, Netlify and GitHub Pages all work with no configuration beyond
build command `npm run build` and output directory `dist`.

## Known limits

- Camera permission is not persisted for installed web apps on iOS, so it asks
  again each launch. That is a WebKit behaviour and the page cannot avoid it.
- Multiplayer is everyone round one phone. There is no networked play and no
  cloud sync; profiles live on the device.
- Cricket is not implemented.
- Vision thresholds want tuning against a real board.
