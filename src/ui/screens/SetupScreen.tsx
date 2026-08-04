import { useState } from 'react'
import { GAME_MODES } from '../../game/modes'
import { useMatch } from '../../store/match'
import type { X01Config } from '../../game/modes/x01'

const MAX_PLAYERS = 8

export function SetupScreen() {
  const startGame = useMatch((s) => s.startGame)
  const [modeId, setModeId] = useState('x01')
  const [names, setNames] = useState<string[]>(['Player 1', 'Player 2'])
  const [startingScore, setStartingScore] = useState(501)
  const [doubleOut, setDoubleOut] = useState(true)
  const [doubleIn, setDoubleIn] = useState(false)
  const [legsToWin, setLegsToWin] = useState(3)

  const mode = GAME_MODES.find((m) => m.id === modeId)!
  const trimmed = names.map((n) => n.trim()).filter(Boolean)
  const duplicate = new Set(trimmed).size !== trimmed.length
  const tooFew = trimmed.length < (modeId === 'killer' ? 2 : 1)
  const canStart = !duplicate && !tooFew

  function start() {
    const config =
      modeId === 'x01'
        ? ({ startingScore, doubleIn, doubleOut, legsToWin, setsToWin: 1 } satisfies X01Config)
        : mode.defaultConfig()
    startGame(modeId, trimmed, config)
  }

  return (
    <div className="mx-auto flex max-w-lg flex-col gap-6 p-5 pb-24">
      <header className="pt-4">
        <h1 className="text-4xl font-black tracking-tight text-amber-400">OCHE</h1>
        <p className="text-sm text-neutral-400">Set the phone up, pick a game, throw.</p>
      </header>

      <section className="flex flex-col gap-2">
        <h2 className="text-xs font-bold uppercase tracking-widest text-neutral-500">Game</h2>
        <div className="flex flex-col gap-2">
          {GAME_MODES.map((m) => (
            <button
              key={m.id}
              type="button"
              onClick={() => setModeId(m.id)}
              className={`rounded-xl border p-3 text-left transition ${
                modeId === m.id
                  ? 'border-amber-400 bg-amber-400/10'
                  : 'border-neutral-800 bg-neutral-900 hover:border-neutral-700'
              }`}
            >
              <div className="font-bold text-neutral-100">{m.name}</div>
              <div className="text-sm text-neutral-400">{m.description}</div>
            </button>
          ))}
        </div>
      </section>

      {modeId === 'x01' && (
        <section className="flex flex-col gap-3">
          <h2 className="text-xs font-bold uppercase tracking-widest text-neutral-500">Rules</h2>
          <div className="flex gap-2">
            {[301, 501, 701].map((value) => (
              <button
                key={value}
                type="button"
                onClick={() => setStartingScore(value)}
                className={`flex-1 rounded-xl py-3 font-bold transition ${
                  startingScore === value
                    ? 'bg-amber-400 text-neutral-950'
                    : 'bg-neutral-800 text-neutral-300'
                }`}
              >
                {value}
              </button>
            ))}
          </div>
          <Toggle label="Double in" checked={doubleIn} onChange={setDoubleIn} />
          <Toggle label="Double out" checked={doubleOut} onChange={setDoubleOut} />
          <label className="flex items-center justify-between rounded-xl bg-neutral-900 px-4 py-3">
            <span className="text-neutral-200">First to</span>
            <div className="flex items-center gap-3">
              <button
                type="button"
                aria-label="Fewer legs"
                onClick={() => setLegsToWin((n) => Math.max(1, n - 1))}
                className="size-9 rounded-lg bg-neutral-800 text-lg font-bold text-neutral-200"
              >
                −
              </button>
              <span className="w-16 text-center font-bold text-neutral-100">
                {legsToWin} leg{legsToWin === 1 ? '' : 's'}
              </span>
              <button
                type="button"
                aria-label="More legs"
                onClick={() => setLegsToWin((n) => Math.min(21, n + 1))}
                className="size-9 rounded-lg bg-neutral-800 text-lg font-bold text-neutral-200"
              >
                +
              </button>
            </div>
          </label>
        </section>
      )}

      <section className="flex flex-col gap-2">
        <h2 className="text-xs font-bold uppercase tracking-widest text-neutral-500">Players</h2>
        {names.map((name, index) => (
          <div key={index} className="flex gap-2">
            <input
              value={name}
              onChange={(e) =>
                setNames((current) => current.map((n, i) => (i === index ? e.target.value : n)))
              }
              aria-label={`Player ${index + 1} name`}
              className="flex-1 rounded-xl border border-neutral-800 bg-neutral-900 px-4 py-3 text-neutral-100 outline-none focus:border-amber-400"
            />
            {names.length > 1 && (
              <button
                type="button"
                aria-label={`Remove player ${index + 1}`}
                onClick={() => setNames((current) => current.filter((_, i) => i !== index))}
                className="rounded-xl bg-neutral-800 px-4 text-neutral-400"
              >
                ✕
              </button>
            )}
          </div>
        ))}
        {names.length < MAX_PLAYERS && (
          <button
            type="button"
            onClick={() => setNames((current) => [...current, `Player ${current.length + 1}`])}
            className="rounded-xl border border-dashed border-neutral-700 py-3 text-neutral-400"
          >
            + Add player
          </button>
        )}
        {duplicate && <p className="text-sm text-red-400">Every player needs a different name.</p>}
      </section>

      <button
        type="button"
        disabled={!canStart}
        onClick={start}
        className="sticky bottom-4 rounded-2xl bg-amber-400 py-4 text-lg font-black text-neutral-950 shadow-lg transition active:scale-[0.98] disabled:opacity-40"
      >
        Start game
      </button>
    </div>
  )
}

function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string
  checked: boolean
  onChange: (value: boolean) => void
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className="flex items-center justify-between rounded-xl bg-neutral-900 px-4 py-3"
    >
      <span className="text-neutral-200">{label}</span>
      <span
        className={`relative h-7 w-12 rounded-full transition ${
          checked ? 'bg-amber-400' : 'bg-neutral-700'
        }`}
      >
        <span
          className={`absolute top-1 size-5 rounded-full bg-white transition-all ${
            checked ? 'left-6' : 'left-1'
          }`}
        />
      </span>
    </button>
  )
}
