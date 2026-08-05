import { useState } from 'react'
import { scoreFromSegment, type BoardScore } from '../../game/board'

const NUMBERS = Array.from({ length: 20 }, (_, i) => i + 1)

export interface KeypadProps {
  onPick: (score: BoardScore) => void
  disabled?: boolean
}

/**
 * Number-pad entry: choose a multiplier, then a number.
 *
 * The multiplier resets to single after each dart, because most darts are
 * singles and a sticky treble is the easiest way to enter a wrong score.
 */
export function Keypad({ onPick, disabled = false }: KeypadProps) {
  const [multiplier, setMultiplier] = useState<1 | 2 | 3>(1)

  function pick(score: BoardScore) {
    onPick(score)
    setMultiplier(1)
  }

  const multiplierButton = (value: 1 | 2 | 3, label: string) => (
    <button
      type="button"
      disabled={disabled}
      onClick={() => setMultiplier((current) => (current === value ? 1 : value))}
      className={`rounded-xl py-3 text-lg font-bold transition ${
        multiplier === value
          ? 'bg-amber-400 text-neutral-950'
          : 'bg-neutral-800 text-neutral-300 hover:bg-neutral-700'
      } disabled:opacity-40`}
    >
      {label}
    </button>
  )

  return (
    <div className="flex flex-col gap-2">
      <div className="grid grid-cols-3 gap-2">
        {multiplierButton(1, 'Single')}
        {multiplierButton(2, 'Double')}
        {multiplierButton(3, 'Treble')}
      </div>

      <div className="grid grid-cols-5 gap-2">
        {NUMBERS.map((n) => (
          <button
            key={n}
            type="button"
            disabled={disabled}
            onClick={() => pick(scoreFromSegment(n, multiplier))}
            className="rounded-xl bg-neutral-800 py-4 text-xl font-semibold text-neutral-100 transition hover:bg-neutral-700 active:scale-95 disabled:opacity-40"
          >
            {n}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-3 gap-2">
        <button
          type="button"
          disabled={disabled}
          onClick={() => pick(scoreFromSegment(25, 1))}
          className="rounded-xl bg-emerald-800 py-4 text-lg font-bold text-emerald-50 transition hover:bg-emerald-700 active:scale-95 disabled:opacity-40"
        >
          25
        </button>
        <button
          type="button"
          disabled={disabled}
          onClick={() => pick(scoreFromSegment(50, 1))}
          className="rounded-xl bg-red-800 py-4 text-lg font-bold text-red-50 transition hover:bg-red-700 active:scale-95 disabled:opacity-40"
        >
          BULL
        </button>
        <button
          type="button"
          disabled={disabled}
          onClick={() => pick(scoreFromSegment(0, 1))}
          className="rounded-xl bg-neutral-700 py-4 text-lg font-bold text-neutral-200 transition hover:bg-neutral-600 active:scale-95 disabled:opacity-40"
        >
          MISS
        </button>
      </div>
    </div>
  )
}
