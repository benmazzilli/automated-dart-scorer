import { useState } from 'react'
import { describeScore, type BoardScore } from '../../game/board'
import { currentPlayer } from '../../game/engine'
import { DARTS_PER_TURN } from '../../game/types'
import { useMatch } from '../../store/match'
import type { Point } from '../../vision/homography'
import { CameraScorer } from '../components/CameraScorer'
import { Dartboard } from '../components/Dartboard'
import { Keypad } from '../components/Keypad'
import { useWakeLock } from '../useWakeLock'

type EntryMode = 'board' | 'keypad' | 'camera'

const ENTRY_LABEL: Record<EntryMode, string> = {
  keypad: 'Keypad',
  board: 'Board',
  camera: 'Camera',
}

/** Cycles keypad → board → camera → keypad. */
const NEXT_ENTRY: Record<EntryMode, EntryMode> = {
  keypad: 'board',
  board: 'camera',
  camera: 'keypad',
}

export function PlayScreen() {
  const mode = useMatch((s) => s.mode)
  const state = useMatch((s) => s.state)
  const throwDart = useMatch((s) => s.throwDart)
  const undo = useMatch((s) => s.undo)
  const undoTurn = useMatch((s) => s.undoTurn)
  const quit = useMatch((s) => s.quit)
  const canUndo = useMatch((s) => s.past.length > 0)
  const nameOf = useMatch((s) => s.nameOf)

  const [entry, setEntry] = useState<EntryMode>('keypad')
  const [correcting, setCorrecting] = useState(false)

  // A phone on a stand gets no touches for minutes, so it would otherwise
  // sleep mid-leg and take the camera with it.
  useWakeLock(true)

  if (!mode || !state) return null

  const player = currentPlayer(state)
  const board = mode.scoreboard(state)
  const target = mode.describeTarget(state)
  const dartsLeft = DARTS_PER_TURN - state.currentTurn.length

  const marks = state.currentTurn
    .filter((t) => t.position)
    .map((t) => ({ ...t.position!, label: describeScore(t.score) }))

  function handlePick(score: BoardScore, position?: { xMm: number; yMm: number }) {
    throwDart(position ? { score, position } : { score })
    // A correction always returns to the board, so leaving it open would hide
    // the camera for the rest of the turn.
    if (correcting) setCorrecting(false)
  }

  function handleCameraScore(
    score: BoardScore,
    position: Point,
    confidence: number,
    corrected: boolean,
  ) {
    throwDart({
      score,
      position: { xMm: position.x, yMm: position.y },
      source: 'camera',
      confidence,
      ...(corrected ? { corrected: true } : {}),
    })
  }

  return (
    <div className="mx-auto flex min-h-dvh max-w-lg flex-col gap-3 p-4 pb-4">
      <header className="flex items-center justify-between">
        <button type="button" onClick={quit} className="text-sm text-neutral-500">
          ← Quit
        </button>
        <span className="text-xs font-bold uppercase tracking-widest text-neutral-500">
          {mode.name} · Round {state.round}
        </span>
      </header>

      <ul className="flex flex-col gap-1.5">
        {board.map((entryRow) => {
          const isThrower = entryRow.playerId === player
          return (
            <li
              key={entryRow.playerId}
              className={`flex items-center justify-between rounded-xl border px-4 py-2.5 transition ${
                isThrower
                  ? 'border-amber-400 bg-amber-400/10'
                  : 'border-neutral-800 bg-neutral-900'
              } ${entryRow.eliminated ? 'opacity-40' : ''}`}
            >
              <div className="min-w-0">
                <div className="truncate font-semibold text-neutral-100">{nameOf(entryRow.playerId)}</div>
                {entryRow.secondary && (
                  <div className="truncate text-xs text-neutral-400">{entryRow.secondary}</div>
                )}
              </div>
              <div
                data-testid={`score-${nameOf(entryRow.playerId)}`}
                className={`shrink-0 pl-3 text-3xl font-black tabular-nums ${
                  isThrower ? 'text-amber-400' : 'text-neutral-300'
                }`}
              >
                {entryRow.primary}
              </div>
            </li>
          )
        })}
      </ul>

      {(target || state.message) && (
        <div className="rounded-xl bg-neutral-900 px-4 py-3 text-center">
          {target && <div className="text-sm text-neutral-400">{target}</div>}
          {state.message && (
            <div
              data-testid="turn-message"
              className="mt-1 text-lg font-black uppercase tracking-wide text-amber-400"
            >
              {state.message}
            </div>
          )}
        </div>
      )}

      <div className="flex items-center justify-center gap-2">
        {Array.from({ length: DARTS_PER_TURN }, (_, i) => {
          const thrown = state.currentTurn[i]
          return (
            <div
              key={i}
              data-testid={`dart-${i}`}
              className={`flex h-12 flex-1 items-center justify-center rounded-xl text-lg font-bold ${
                thrown
                  ? 'bg-neutral-100 text-neutral-900'
                  : 'border border-dashed border-neutral-700 text-neutral-700'
              }`}
            >
              {thrown ? describeScore(thrown.score) : '–'}
            </div>
          )
        })}
      </div>

      {correcting ? (
        <div className="flex flex-col gap-2">
          <p className="text-center text-sm text-neutral-400">
            Tap where the dart actually landed
          </p>
          <Dartboard onPick={handlePick} marks={marks} className="mx-auto w-full max-w-sm" />
          <button
            type="button"
            onClick={() => setCorrecting(false)}
            className="rounded-xl bg-neutral-800 py-3 text-sm font-semibold text-neutral-200"
          >
            Cancel
          </button>
        </div>
      ) : entry === 'camera' ? (
        <CameraScorer
          onScore={handleCameraScore}
          onCorrect={() => setCorrecting(true)}
          disabled={dartsLeft === 0}
        />
      ) : entry === 'board' ? (
        <Dartboard
          onPick={handlePick}
          marks={marks}
          className="mx-auto w-full max-w-sm"
          disabled={dartsLeft === 0}
        />
      ) : (
        <Keypad onPick={(score) => handlePick(score)} disabled={dartsLeft === 0} />
      )}

      <div className="mt-auto grid grid-cols-3 gap-2 pt-2">
        <button
          type="button"
          onClick={() => setEntry((m) => NEXT_ENTRY[m])}
          className="rounded-xl bg-neutral-800 py-3 text-sm font-semibold text-neutral-200"
        >
          {ENTRY_LABEL[NEXT_ENTRY[entry]]}
        </button>
        <button
          type="button"
          onClick={undo}
          disabled={!canUndo}
          className="rounded-xl bg-neutral-800 py-3 text-sm font-semibold text-neutral-200 disabled:opacity-40"
        >
          Undo dart
        </button>
        <button
          type="button"
          onClick={undoTurn}
          disabled={!canUndo}
          className="rounded-xl bg-neutral-800 py-3 text-sm font-semibold text-neutral-200 disabled:opacity-40"
        >
          Undo turn
        </button>
      </div>
    </div>
  )
}
