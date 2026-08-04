# Your own memes

The app ships with original built-in effects — big text, screen shake,
confetti, colour flash. This folder is where you add your own images and
sounds on top, without rebuilding anything.

## Adding one

1. Drop an image or sound in this folder, e.g. `pointing-rick.png`,
   `airhorn.mp3`.
2. Add an entry to `memes.config.json` naming the file and what fires it.
3. Reload the app. That's it — the config is fetched at runtime.

## Config

```json
{
  "replaceDefaults": false,
  "memes": [
    {
      "id": "my-180",
      "trigger": { "on": "turnScore", "op": "eq", "value": 180 },
      "text": "GET IN",
      "image": "pointing-rick.png",
      "sound": "airhorn.mp3",
      "effect": "confetti",
      "colour": "#fbbf24",
      "durationMs": 2600,
      "priority": 150
    }
  ]
}
```

Set `replaceDefaults` to `true` to throw away the built-ins entirely and use
only your own. Give an entry the same `id` as a built-in to replace just that
one.

## Triggers

| Trigger | Fires when |
| --- | --- |
| `{ "on": "turnScore", "op": "eq" \| "gte" \| "lte", "value": n }` | A visit scores exactly / at least / at most `n` |
| `{ "on": "bust" }` | A visit busts |
| `{ "on": "checkout", "minValue": n }` | A leg is won, optionally only from `n` or above |
| `{ "on": "nineDarter" }` | A leg won in nine darts |
| `{ "on": "missedDouble" }` | A finish was on at the start of the visit and was not taken |
| `{ "on": "gameWon" }` | The match ends |
| `{ "on": "whitewash" }` | The match ends with the loser on no legs |

## Effects

`slam` (scales in), `shake`, `confetti`, `flash`, `none`.

`priority` decides which one wins when several match the same moment — highest
takes it. The built-ins run from 30 (a poor visit) to 210 (whitewash), so pick
above 210 if you want yours to always win.

## Notes

- Everything in this folder except this README and `memes.config.json` is
  gitignored, so your assets stay yours and out of the repository.
- Sounds may not play until you have tapped the screen once — browsers block
  autoplay before an interaction. The visual effect always plays.
- Assets are deliberately excluded from the service worker precache, so a
  folder full of large gifs will not bloat the offline install.
