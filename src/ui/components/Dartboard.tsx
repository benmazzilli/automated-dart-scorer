import { useMemo, useRef } from 'react'
import {
  RADIUS,
  SEGMENTS,
  SEGMENT_ANGLE,
  polarToCartesian,
  scoreFromCartesian,
  type BoardScore,
} from '../../game/board'

/** The board is drawn in millimetres and scaled by the SVG viewBox. */
const VIEW = RADIUS.doubleOuter * 1.16

const COLOURS = {
  surround: '#12100e',
  wire: '#d8d4cc',
  darkBed: '#141414',
  lightBed: '#e8dcc0',
  redBed: '#c1272d',
  greenBed: '#1c7a4a',
  numbers: '#f4f1ea',
}

interface Ring {
  inner: number
  outer: number
  /** Beds alternate between these two, starting with the 20. */
  even: string
  odd: string
}

const RINGS: Ring[] = [
  { inner: RADIUS.outerBull, outer: RADIUS.trebleInner, even: COLOURS.darkBed, odd: COLOURS.lightBed },
  { inner: RADIUS.trebleInner, outer: RADIUS.trebleOuter, even: COLOURS.redBed, odd: COLOURS.greenBed },
  { inner: RADIUS.trebleOuter, outer: RADIUS.doubleInner, even: COLOURS.darkBed, odd: COLOURS.lightBed },
  { inner: RADIUS.doubleInner, outer: RADIUS.doubleOuter, even: COLOURS.redBed, odd: COLOURS.greenBed },
]

/** An annular sector path, in board millimetres. */
function sectorPath(inner: number, outer: number, fromDeg: number, toDeg: number): string {
  const a = polarToCartesian(outer, fromDeg)
  const b = polarToCartesian(outer, toDeg)
  const c = polarToCartesian(inner, toDeg)
  const d = polarToCartesian(inner, fromDeg)
  const largeArc = toDeg - fromDeg > 180 ? 1 : 0
  return [
    `M ${a.xMm} ${a.yMm}`,
    `A ${outer} ${outer} 0 ${largeArc} 1 ${b.xMm} ${b.yMm}`,
    `L ${c.xMm} ${c.yMm}`,
    `A ${inner} ${inner} 0 ${largeArc} 0 ${d.xMm} ${d.yMm}`,
    'Z',
  ].join(' ')
}

export interface DartboardProps {
  /** Called with the score and board position when a point is tapped. */
  onPick?: (score: BoardScore, position: { xMm: number; yMm: number }) => void
  /** Markers to draw, e.g. the darts thrown this turn. */
  marks?: { xMm: number; yMm: number; label?: string }[]
  /** Dim the board and ignore taps. */
  disabled?: boolean
  className?: string
}

/**
 * An interactive dartboard rendered from the same geometry constants the
 * scoring uses, so what you tap is exactly what gets scored — there is no
 * second copy of the layout to drift out of step.
 */
export function Dartboard({ onPick, marks = [], disabled = false, className }: DartboardProps) {
  const svgRef = useRef<SVGSVGElement>(null)

  const beds = useMemo(() => {
    const paths: { d: string; fill: string; key: string }[] = []
    SEGMENTS.forEach((value, index) => {
      const from = index * SEGMENT_ANGLE - SEGMENT_ANGLE / 2
      const to = from + SEGMENT_ANGLE
      for (const [ringIndex, ring] of RINGS.entries()) {
        paths.push({
          key: `${value}-${ringIndex}`,
          d: sectorPath(ring.inner, ring.outer, from, to),
          fill: index % 2 === 0 ? ring.even : ring.odd,
        })
      }
    })
    return paths
  }, [])

  const numbers = useMemo(
    () =>
      SEGMENTS.map((value, index) => {
        const { xMm, yMm } = polarToCartesian(RADIUS.doubleOuter * 1.09, index * SEGMENT_ANGLE)
        return { value, xMm, yMm }
      }),
    [],
  )

  function handlePointer(event: React.PointerEvent<SVGSVGElement>) {
    if (disabled || !onPick) return
    const svg = svgRef.current
    if (!svg) return

    // Map the click through the SVG's own transform so the result is correct
    // whatever size the board is rendered at.
    const point = svg.createSVGPoint()
    point.x = event.clientX
    point.y = event.clientY
    const matrix = svg.getScreenCTM()
    if (!matrix) return
    const local = point.matrixTransform(matrix.inverse())

    onPick(scoreFromCartesian(local.x, local.y), { xMm: local.x, yMm: local.y })
  }

  return (
    <svg
      ref={svgRef}
      viewBox={`${-VIEW} ${-VIEW} ${VIEW * 2} ${VIEW * 2}`}
      className={className}
      onPointerDown={handlePointer}
      role={onPick ? 'button' : 'img'}
      aria-label={onPick ? 'Dartboard — tap where the dart landed' : 'Dartboard'}
      style={{ touchAction: 'manipulation', opacity: disabled ? 0.45 : 1 }}
    >
      <circle cx={0} cy={0} r={VIEW} fill={COLOURS.surround} />

      {beds.map((bed) => (
        <path key={bed.key} d={bed.d} fill={bed.fill} />
      ))}

      {/* Wires drawn over the beds, matching the real spider. */}
      <g stroke={COLOURS.wire} strokeWidth={0.7} fill="none" opacity={0.55}>
        {SEGMENTS.map((_, index) => {
          const angle = index * SEGMENT_ANGLE - SEGMENT_ANGLE / 2
          const from = polarToCartesian(RADIUS.outerBull, angle)
          const to = polarToCartesian(RADIUS.doubleOuter, angle)
          return <line key={index} x1={from.xMm} y1={from.yMm} x2={to.xMm} y2={to.yMm} />
        })}
        {[RADIUS.trebleInner, RADIUS.trebleOuter, RADIUS.doubleInner, RADIUS.doubleOuter].map((r) => (
          <circle key={r} cx={0} cy={0} r={r} />
        ))}
      </g>

      <circle cx={0} cy={0} r={RADIUS.outerBull} fill={COLOURS.greenBed} />
      <circle cx={0} cy={0} r={RADIUS.innerBull} fill={COLOURS.redBed} />
      <g stroke={COLOURS.wire} strokeWidth={0.7} fill="none" opacity={0.55}>
        <circle cx={0} cy={0} r={RADIUS.outerBull} />
        <circle cx={0} cy={0} r={RADIUS.innerBull} />
      </g>

      {numbers.map(({ value, xMm, yMm }) => (
        <text
          key={value}
          x={xMm}
          y={yMm}
          fill={COLOURS.numbers}
          fontSize={17}
          fontWeight={700}
          textAnchor="middle"
          dominantBaseline="central"
          style={{ pointerEvents: 'none', userSelect: 'none' }}
        >
          {value}
        </text>
      ))}

      {marks.map((mark, index) => (
        <g key={index} style={{ pointerEvents: 'none' }}>
          <circle cx={mark.xMm} cy={mark.yMm} r={5.5} fill="#fbbf24" stroke="#111" strokeWidth={1.5} />
          {mark.label && (
            <text
              x={mark.xMm}
              y={mark.yMm - 11}
              fill="#fbbf24"
              fontSize={12}
              fontWeight={700}
              textAnchor="middle"
            >
              {mark.label}
            </text>
          )}
        </g>
      ))}
    </svg>
  )
}
