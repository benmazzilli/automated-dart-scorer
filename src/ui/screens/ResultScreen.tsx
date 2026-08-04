import { useMatch } from '../../store/match'

const PLACES = ['🥇', '🥈', '🥉']

export function ResultScreen() {
  const mode = useMatch((s) => s.mode)
  const state = useMatch((s) => s.state)
  const quit = useMatch((s) => s.quit)
  const startGame = useMatch((s) => s.startGame)
  const profiles = useMatch((s) => s.profiles)
  const nameOf = useMatch((s) => s.nameOf)

  if (!mode || !state) return null

  const rankings = state.rankings.length > 0 ? state.rankings : state.players

  return (
    <div className="mx-auto flex min-h-dvh max-w-lg flex-col justify-center gap-6 p-6">
      <div className="text-center">
        <p className="text-xs font-bold uppercase tracking-widest text-neutral-500">{mode.name}</p>
        <h1 className="mt-2 text-5xl font-black text-amber-400">
          {state.winner ? `${nameOf(state.winner)} wins` : 'Draw'}
        </h1>
        {state.message && <p className="mt-2 text-neutral-400">{state.message}</p>}
      </div>

      <ol className="flex flex-col gap-2">
        {rankings.map((playerId, index) => (
          <li
            key={playerId}
            className={`flex items-center gap-3 rounded-xl border px-4 py-3 ${
              index === 0 ? 'border-amber-400 bg-amber-400/10' : 'border-neutral-800 bg-neutral-900'
            }`}
          >
            <span className="w-8 text-center text-lg">{PLACES[index] ?? index + 1}</span>
            <span className="flex-1 font-semibold text-neutral-100">{nameOf(playerId)}</span>
          </li>
        ))}
      </ol>

      <div className="flex flex-col gap-2">
        <button
          type="button"
          onClick={() => startGame(mode.id, profiles, undefined)}
          className="rounded-2xl bg-amber-400 py-4 text-lg font-black text-neutral-950"
        >
          Play again
        </button>
        <button
          type="button"
          onClick={quit}
          className="rounded-2xl bg-neutral-800 py-4 font-semibold text-neutral-200"
        >
          New game
        </button>
      </div>
    </div>
  )
}
