import { useState } from 'react'
import { useLiveQuery } from 'dexie-react-hooks'
import { GAME_MODES } from '../../game/modes'
import { useMatch } from '../../store/match'
import type { X01Config } from '../../game/modes/x01'
import { createProfile, listProfiles } from '../../db/queries'
import type { Profile } from '../../db/schema'

const MAX_PLAYERS = 8

export function SetupScreen({ onShowStats }: { onShowStats?: () => void }) {
  const startGame = useMatch((s) => s.startGame)
  const [modeId, setModeId] = useState('x01')
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const [newName, setNewName] = useState('')
  const [startingScore, setStartingScore] = useState(501)
  const [doubleOut, setDoubleOut] = useState(true)
  const [doubleIn, setDoubleIn] = useState(false)
  const [legsToWin, setLegsToWin] = useState(3)

  const profiles = useLiveQuery(listProfiles, [], [] as Profile[])

  const mode = GAME_MODES.find((m) => m.id === modeId)!
  // Selection order is the throwing order, so preserve it rather than the
  // order profiles happen to be stored in.
  const selected = selectedIds
    .map((id) => profiles.find((p) => p.id === id))
    .filter((p): p is Profile => p !== undefined)
  const canStart = selected.length >= (modeId === 'killer' ? 2 : 1)

  function toggle(id: string) {
    setSelectedIds((current) =>
      current.includes(id)
        ? current.filter((x) => x !== id)
        : current.length >= MAX_PLAYERS
          ? current
          : [...current, id],
    )
  }

  async function addPlayer() {
    const name = newName.trim()
    if (!name) return
    // Clear the field before awaiting the write, not after. Clearing
    // afterwards wipes whatever was typed while the database round-trip was in
    // flight, which loses the next player's name entirely.
    setNewName('')
    const profile = await createProfile(name)
    setSelectedIds((current) => (current.includes(profile.id) ? current : [...current, profile.id]))
  }

  function start() {
    const config =
      modeId === 'x01'
        ? ({ startingScore, doubleIn, doubleOut, legsToWin, setsToWin: 1 } satisfies X01Config)
        : mode.defaultConfig()
    startGame(modeId, selected, config)
  }

  return (
    <div className="mx-auto flex max-w-lg flex-col gap-6 p-5 pb-24">
      <header className="flex items-end justify-between pt-4">
        <div>
          <h1 className="text-4xl font-black tracking-tight text-amber-400">OCHE</h1>
          <p className="text-sm text-neutral-400">Set the phone up, pick a game, throw.</p>
        </div>
        {onShowStats && (
          <button
            type="button"
            onClick={onShowStats}
            className="rounded-xl bg-neutral-800 px-4 py-2 text-sm font-semibold text-neutral-200"
          >
            Stats
          </button>
        )}
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
        <h2 className="text-xs font-bold uppercase tracking-widest text-neutral-500">
          Players {selected.length > 0 && `· throwing in this order`}
        </h2>

        {profiles.length === 0 && (
          <p className="text-sm text-neutral-500">
            No players yet. Add everyone who is throwing — their stats are kept between games.
          </p>
        )}

        <div className="flex flex-wrap gap-2">
          {profiles.map((profile) => {
            const order = selectedIds.indexOf(profile.id)
            const isSelected = order !== -1
            return (
              <button
                key={profile.id}
                type="button"
                aria-pressed={isSelected}
                data-testid={`player-${profile.name}`}
                onClick={() => toggle(profile.id)}
                className={`flex items-center gap-2 rounded-full border px-4 py-2.5 font-semibold transition ${
                  isSelected
                    ? 'border-transparent text-neutral-950'
                    : 'border-neutral-700 bg-neutral-900 text-neutral-300'
                }`}
                style={isSelected ? { backgroundColor: profile.colour } : undefined}
              >
                {isSelected && (
                  <span className="grid size-5 place-items-center rounded-full bg-black/25 text-xs">
                    {order + 1}
                  </span>
                )}
                {profile.name}
              </button>
            )
          })}
        </div>

        <div className="flex gap-2">
          <input
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') void addPlayer()
            }}
            placeholder="Add a player…"
            aria-label="New player name"
            className="flex-1 rounded-xl border border-neutral-800 bg-neutral-900 px-4 py-3 text-neutral-100 outline-none placeholder:text-neutral-600 focus:border-amber-400"
          />
          <button
            type="button"
            onClick={() => void addPlayer()}
            disabled={!newName.trim()}
            className="rounded-xl bg-neutral-800 px-5 font-semibold text-neutral-200 disabled:opacity-40"
          >
            Add
          </button>
        </div>

        {modeId === 'killer' && selected.length < 2 && (
          <p className="text-sm text-neutral-500">Killer needs at least two players.</p>
        )}
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
