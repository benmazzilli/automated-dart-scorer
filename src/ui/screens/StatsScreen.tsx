import { useState } from 'react'
import { useLiveQuery } from 'dexie-react-hooks'
import { deleteProfile, listMatches, listProfiles } from '../../db/queries'
import type { Profile } from '../../db/schema'
import { getMode } from '../../game/modes'
import {
  averageTrend,
  computeStats,
  formatAverage,
  formatRate,
  headToHead,
  type MatchRecord,
} from '../../game/stats'

export function StatsScreen({ onBack }: { onBack: () => void }) {
  const profiles = useLiveQuery(listProfiles, [], [] as Profile[])
  const matches = useLiveQuery(() => listMatches(), [], [] as MatchRecord[])
  const [selectedId, setSelectedId] = useState<string | null>(null)

  const selected = profiles.find((p) => p.id === selectedId) ?? profiles[0] ?? null
  const nameOf = (id: string) => profiles.find((p) => p.id === id)?.name ?? 'Unknown'

  const stats = selected ? computeStats(selected.id, matches) : null
  const h2h = selected ? headToHead(selected.id, matches) : []
  const trend = selected ? averageTrend(selected.id, matches) : []

  return (
    <div className="mx-auto flex min-h-dvh max-w-lg flex-col gap-5 p-5 pb-10">
      <header className="flex items-center justify-between pt-2">
        <button type="button" onClick={onBack} className="text-sm text-neutral-500">
          ← Back
        </button>
        <h1 className="text-lg font-black tracking-tight text-amber-400">STATS</h1>
      </header>

      {profiles.length === 0 && (
        <p className="text-neutral-500">No players yet. Add one on the setup screen and play a game.</p>
      )}

      <div className="flex flex-wrap gap-2">
        {profiles.map((profile) => (
          <button
            key={profile.id}
            type="button"
            data-testid={`profile-${profile.name}`}
            onClick={() => setSelectedId(profile.id)}
            className={`rounded-full border px-4 py-2 font-semibold transition ${
              selected?.id === profile.id
                ? 'border-transparent text-neutral-950'
                : 'border-neutral-700 bg-neutral-900 text-neutral-300'
            }`}
            style={selected?.id === profile.id ? { backgroundColor: profile.colour } : undefined}
          >
            {profile.name}
          </button>
        ))}
      </div>

      {selected && stats && (
        <>
          {stats.matchesPlayed === 0 ? (
            <p className="text-neutral-500">{selected.name} has not finished a game yet.</p>
          ) : (
            <>
              <section className="grid grid-cols-2 gap-2">
                <Stat label="Played" value={String(stats.matchesPlayed)} />
                <Stat label="Won" value={`${stats.matchesWon} · ${formatRate(stats.winRate)}`} />
                <Stat label="3-dart average" value={formatAverage(stats.threeDartAverage)} />
                <Stat label="First 9" value={formatAverage(stats.firstNineAverage)} />
                <Stat label="Checkout" value={formatRate(stats.checkoutRate)} highlight />
                <Stat
                  label="Best leg"
                  value={stats.bestLegDarts === null ? '—' : `${stats.bestLegDarts} darts`}
                />
                <Stat label="180s" value={String(stats.count180)} />
                <Stat label="140+" value={String(stats.count140plus)} />
                <Stat label="100+" value={String(stats.count100plus)} />
                <Stat label="Highest turn" value={String(stats.highestTurn)} />
                <Stat label="Best finish" value={stats.highestCheckout ? String(stats.highestCheckout) : '—'} />
                <Stat label="Busts" value={String(stats.busts)} />
              </section>

              {trend.length > 1 && <Sparkline points={trend.map((p) => p.average)} />}

              {h2h.length > 0 && (
                <section className="flex flex-col gap-2">
                  <h2 className="text-xs font-bold uppercase tracking-widest text-neutral-500">
                    Head to head
                  </h2>
                  {h2h.map((row) => (
                    <div
                      key={row.opponentId}
                      className="flex items-center justify-between rounded-xl bg-neutral-900 px-4 py-3"
                    >
                      <span className="text-neutral-200">vs {nameOf(row.opponentId)}</span>
                      <span className="font-bold tabular-nums text-neutral-100">
                        <span className="text-emerald-400">{row.won}</span>
                        {' – '}
                        <span className="text-red-400">{row.lost}</span>
                      </span>
                    </div>
                  ))}
                </section>
              )}
            </>
          )}

          <RecentMatches matches={matches.filter((m) => m.playerIds.includes(selected.id))} nameOf={nameOf} />

          <button
            type="button"
            onClick={() => {
              if (confirm(`Delete ${selected.name}? Their past games are kept.`)) {
                void deleteProfile(selected.id)
                setSelectedId(null)
              }
            }}
            className="mt-2 rounded-xl border border-red-900/60 py-3 text-sm font-semibold text-red-400"
          >
            Delete {selected.name}
          </button>
        </>
      )}
    </div>
  )
}

function Stat({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div data-testid={`stat-${label}`} className="rounded-xl bg-neutral-900 px-4 py-3">
      <div className="text-xs uppercase tracking-wide text-neutral-500">{label}</div>
      <div className={`text-2xl font-black tabular-nums ${highlight ? 'text-amber-400' : 'text-neutral-100'}`}>
        {value}
      </div>
    </div>
  )
}

/** A minimal trend line for three-dart averages over time. */
function Sparkline({ points }: { points: number[] }) {
  const max = Math.max(...points)
  const min = Math.min(...points)
  const span = max - min || 1
  const path = points
    .map((value, index) => {
      const x = (index / (points.length - 1)) * 100
      const y = 30 - ((value - min) / span) * 26
      return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`
    })
    .join(' ')

  return (
    <section className="rounded-xl bg-neutral-900 p-4">
      <h2 className="text-xs font-bold uppercase tracking-widest text-neutral-500">
        Average per game
      </h2>
      <svg viewBox="0 0 100 32" preserveAspectRatio="none" className="mt-2 h-16 w-full">
        <path d={path} fill="none" stroke="#fbbf24" strokeWidth={1.2} vectorEffect="non-scaling-stroke" />
      </svg>
      <div className="flex justify-between text-xs text-neutral-500">
        <span>{formatAverage(min)}</span>
        <span>{formatAverage(max)}</span>
      </div>
    </section>
  )
}

function RecentMatches({
  matches,
  nameOf,
}: {
  matches: MatchRecord[]
  nameOf: (id: string) => string
}) {
  if (matches.length === 0) return null
  return (
    <section className="flex flex-col gap-2">
      <h2 className="text-xs font-bold uppercase tracking-widest text-neutral-500">Recent games</h2>
      {matches.slice(0, 10).map((match) => (
        <div key={match.id} className="flex items-center justify-between rounded-xl bg-neutral-900 px-4 py-3">
          <div className="min-w-0">
            <div className="truncate text-sm text-neutral-200">
              {safeModeName(match.modeId)} · {match.playerIds.map(nameOf).join(' v ')}
            </div>
            <div className="text-xs text-neutral-500">
              {new Date(match.finishedAt).toLocaleDateString()}
            </div>
          </div>
          <span className="shrink-0 pl-3 text-sm font-bold text-amber-400">
            {match.winnerId ? nameOf(match.winnerId) : 'Draw'}
          </span>
        </div>
      ))}
    </section>
  )
}

/** Match records can outlive a mode being renamed or removed. */
function safeModeName(modeId: string): string {
  try {
    return getMode(modeId).name
  } catch {
    return modeId
  }
}
