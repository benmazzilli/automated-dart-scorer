import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { RADIUS, describeScore, polarToCartesian, type BoardScore } from '../../game/board'
import { useVision } from '../../store/vision'
import {
  CALIBRATION_TARGETS,
  boardPointToImage,
  calibrate,
  calibrationLooksSane,
  type Calibration,
  type CalibrationPoints,
} from '../../vision/calibration'
import type { GrayImage, Mask } from '../../vision/frameDiff'
import type { Point } from '../../vision/homography'
import { ThrowWatcher, type DartReading, type WatcherPhase } from '../../vision/pipeline'
import { useCamera } from '../../vision/useCamera'

export interface CameraScorerProps {
  /** Called when a reading is accepted, automatically or by hand. */
  onScore: (score: BoardScore, position: Point, confidence: number, corrected: boolean) => void
  /** Opens the board diagram so a wrong reading can be put right. */
  onCorrect: (reading: DartReading) => void
  disabled?: boolean
}

/**
 * Live camera scoring: calibrate once, then darts are read as they land.
 *
 * Every reading is shown before it counts. A confident one is accepted after a
 * short pause; anything less waits to be confirmed. That is not a hedge — a
 * single camera cannot resolve the depth of a dart pointing at the lens, so
 * being easy to correct matters more than being right every time.
 */
export function CameraScorer({ onScore, onCorrect, disabled = false }: CameraScorerProps) {
  const taps = useVision((s) => s.taps)
  const setTaps = useVision((s) => s.setTaps)
  const config = useVision((s) => s.config)
  const debug = useVision((s) => s.debug)
  const toggleDebug = useVision((s) => s.toggleDebug)

  const calibration = useMemo(() => (taps ? calibrate(taps) : null), [taps])
  const [pending, setPending] = useState<Point[]>([])
  const [phase, setPhase] = useState<WatcherPhase>('waiting')
  const [reading, setReading] = useState<DartReading | null>(null)
  const [lastMask, setLastMask] = useState<Mask | null>(null)

  const watcherRef = useRef<ThrowWatcher | null>(null)
  const previewRef = useRef<HTMLDivElement | null>(null)
  const overlayRef = useRef<HTMLCanvasElement | null>(null)

  // Readings waiting to be shown, and the one on screen now.
  //
  // A queue rather than a single slot, because darts land faster than a person
  // reads a confirmation. Holding one reading in state and replacing it when
  // the next arrives loses every dart but the last — throw three in quick
  // succession and only the third would score.
  const queueRef = useRef<DartReading[]>([])
  const currentRef = useRef<DartReading | null>(null)
  const onScoreRef = useRef(onScore)
  useEffect(() => {
    onScoreRef.current = onScore
  }, [onScore])

  const calibrating = calibration === null

  const advance = useCallback(() => {
    const next = queueRef.current.shift() ?? null
    currentRef.current = next
    setReading(next)
  }, [])

  const commit = useCallback(
    (accepted: DartReading, corrected: boolean) => {
      onScoreRef.current(accepted.score, accepted.boardPoint, accepted.confidence, corrected)
      advance()
    },
    [advance],
  )

  useEffect(() => {
    watcherRef.current = calibration ? new ThrowWatcher(calibration, config) : null
  }, [calibration, config])

  const handleFrame = useCallback(
    (frame: GrayImage) => {
      const watcher = watcherRef.current
      if (!watcher || disabled) return

      const update = watcher.push(frame)
      setPhase(update.phase)

      if (update.reading) {
        if (debug) {
          console.info('[oche] dart read', {
            score: describeScore(update.reading.score),
            board: update.reading.boardPoint,
            confidence: update.reading.confidence,
            area: update.reading.blob.area,
          })
        }
        queueRef.current.push(update.reading)
        if (!currentRef.current) advance()
      }

      if (update.failure && debug) console.info('[oche] no dart read:', update.failure)
      if (debug && update.reading) setLastMask(update.reading.mask)
    },
    [disabled, debug, advance],
  )

  const { videoRef, state } = useCamera({
    width: config.frameWidth,
    height: config.frameHeight,
    fps: config.captureFps,
    onFrame: handleFrame,
    enabled: true,
  })

  // Accept a confident reading after a beat, so a good throw needs no taps.
  // When darts are queued behind it, hurry up rather than falling further
  // behind the player.
  useEffect(() => {
    if (!reading) return
    if (reading.confidence < config.autoAcceptConfidence) return
    const delay = queueRef.current.length > 0 ? 350 : 1500
    const timer = setTimeout(() => commit(reading, false), delay)
    return () => clearTimeout(timer)
  }, [reading, config.autoAcceptConfidence, commit])

  // Draw the debug overlay.
  useEffect(() => {
    const canvas = overlayRef.current
    if (!canvas || !debug) return
    const context = canvas.getContext('2d')
    if (!context) return
    context.clearRect(0, 0, canvas.width, canvas.height)
    if (!lastMask) return

    const image = context.createImageData(lastMask.width, lastMask.height)
    for (let i = 0; i < lastMask.data.length; i++) {
      if (lastMask.data[i] === 1) {
        image.data[i * 4] = 251
        image.data[i * 4 + 1] = 191
        image.data[i * 4 + 2] = 36
        image.data[i * 4 + 3] = 190
      }
    }
    context.putImageData(image, 0, 0)
  }, [lastMask, debug])

  /** A tap on the preview, in the frame's pixel coordinates. */
  function tapPoint(event: React.PointerEvent<HTMLDivElement>): Point | null {
    const box = previewRef.current?.getBoundingClientRect()
    if (!box || box.width === 0) return null
    return {
      x: ((event.clientX - box.left) / box.width) * config.frameWidth,
      y: ((event.clientY - box.top) / box.height) * config.frameHeight,
    }
  }

  function handleCalibrationTap(event: React.PointerEvent<HTMLDivElement>) {
    const point = tapPoint(event)
    if (!point) return

    const next = [...pending, point]
    if (next.length < 4) {
      setPending(next)
      return
    }

    const candidate = calibrate(next as unknown as CalibrationPoints)
    setPending([])
    if (candidate && calibrationLooksSane(candidate)) {
      setTaps(next as unknown as CalibrationPoints)
    } else {
      // Silently keeping a nonsense calibration would score every dart wrong
      // with no clue why, so start over instead.
      window.alert(
        'Those taps do not make a board. Tap the outer edge of the double ring at the 20, then the 6, then the 3, then the 11.',
      )
    }
  }

  const target = CALIBRATION_TARGETS[pending.length]

  return (
    <div className="flex flex-col gap-2">
      <div
        ref={previewRef}
        onPointerDown={calibrating ? handleCalibrationTap : undefined}
        className="relative aspect-[4/3] w-full overflow-hidden rounded-xl bg-black"
        style={{ touchAction: 'manipulation' }}
      >
        <video
          ref={videoRef}
          playsInline
          muted
          autoPlay
          className="size-full object-cover"
        />

        {debug && (
          <canvas
            ref={overlayRef}
            width={config.frameWidth}
            height={config.frameHeight}
            className="pointer-events-none absolute inset-0 size-full object-cover opacity-80"
          />
        )}

        {calibration && !calibrating && (
          <BoardGuide calibration={calibration} config={{ w: config.frameWidth, h: config.frameHeight }} />
        )}

        {calibrating && <CalibrationMarks points={pending} config={config} />}

        {state.status !== 'live' && (
          <div className="absolute inset-0 grid place-items-center bg-black/80 p-6 text-center">
            <p className="text-sm text-neutral-300">
              {state.message ??
                (state.status === 'starting' ? 'Starting the camera…' : 'Camera not running.')}
            </p>
          </div>
        )}

        {state.status === 'live' && calibrating && (
          <div className="absolute inset-x-0 bottom-0 bg-black/75 p-3 text-center">
            <p className="text-xs uppercase tracking-widest text-neutral-400">
              Calibrate · {pending.length + 1} of 4
            </p>
            <p className="font-bold text-amber-400">{target?.label}</p>
            <p className="text-xs text-neutral-400">Tap the outer edge of the double ring</p>
          </div>
        )}
      </div>

      {!calibrating && (
        <div className="flex items-center justify-between rounded-xl bg-neutral-900 px-4 py-2">
          <span className="text-xs uppercase tracking-widest text-neutral-500">
            {phase === 'moving' ? 'Movement…' : phase === 'settling' ? 'Settling…' : 'Watching'}
          </span>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={toggleDebug}
              className={`rounded-lg px-3 py-1.5 text-xs font-semibold ${
                debug ? 'bg-amber-400 text-neutral-950' : 'bg-neutral-800 text-neutral-300'
              }`}
            >
              Debug
            </button>
            <button
              type="button"
              onClick={() => setTaps(null)}
              className="rounded-lg bg-neutral-800 px-3 py-1.5 text-xs font-semibold text-neutral-300"
            >
              Recalibrate
            </button>
          </div>
        </div>
      )}

      {reading && (
        <div className="flex items-center gap-3 rounded-xl border border-amber-400 bg-amber-400/10 px-4 py-3">
          <div className="flex-1">
            <div className="text-xs uppercase tracking-widest text-neutral-400">
              {reading.confidence >= config.autoAcceptConfidence ? 'Scored' : 'Not sure — check it'}
            </div>
            <div className="text-3xl font-black text-amber-400">
              {describeScore(reading.score)}
            </div>
          </div>
          <button
            type="button"
            onClick={() => {
              onCorrect(reading)
              advance()
            }}
            className="rounded-xl bg-neutral-800 px-4 py-3 text-sm font-semibold text-neutral-200"
          >
            Wrong
          </button>
          <button
            type="button"
            onClick={() => commit(reading, false)}
            className="rounded-xl bg-amber-400 px-5 py-3 text-sm font-black text-neutral-950"
          >
            OK
          </button>
        </div>
      )}
    </div>
  )
}

/** The rings drawn over the preview, so a bad calibration is obvious at a glance. */
function BoardGuide({
  calibration,
  config,
}: {
  calibration: Calibration
  config: { w: number; h: number }
}) {
  const rings = [RADIUS.doubleOuter, RADIUS.trebleOuter, RADIUS.outerBull]
  const paths = rings.map((radius) => {
    const points: string[] = []
    for (let bearing = 0; bearing <= 360; bearing += 6) {
      const mm = polarToCartesian(radius, bearing)
      const image = boardPointToImage(calibration, { x: mm.xMm, y: mm.yMm })
      points.push(`${points.length === 0 ? 'M' : 'L'} ${image.x.toFixed(1)} ${image.y.toFixed(1)}`)
    }
    return points.join(' ') + ' Z'
  })

  return (
    <svg
      viewBox={`0 0 ${config.w} ${config.h}`}
      className="pointer-events-none absolute inset-0 size-full object-cover"
    >
      {paths.map((d, i) => (
        <path key={i} d={d} fill="none" stroke="#fbbf24" strokeWidth={1.5} opacity={0.4} />
      ))}
    </svg>
  )
}

function CalibrationMarks({
  points,
  config,
}: {
  points: Point[]
  config: { frameWidth: number; frameHeight: number }
}) {
  return (
    <svg
      viewBox={`0 0 ${config.frameWidth} ${config.frameHeight}`}
      className="pointer-events-none absolute inset-0 size-full object-cover"
    >
      {points.map((point, index) => (
        <g key={index}>
          <circle cx={point.x} cy={point.y} r={9} fill="none" stroke="#fbbf24" strokeWidth={2.5} />
          <circle cx={point.x} cy={point.y} r={2.5} fill="#fbbf24" />
          <text x={point.x + 13} y={point.y + 5} fill="#fbbf24" fontSize={16} fontWeight={700}>
            {index + 1}
          </text>
        </g>
      ))}
    </svg>
  )
}
