import { useState, type CSSProperties, type PointerEvent } from 'react'
import { YOUS, youById } from '../yous'
import './LearningCurve.css'

const W = 640
const H = 300
const M = { top: 14, right: 96, bottom: 34, left: 46 }
const LABEL_GAP = 21

const pct = (x: number) => `${Math.round(x * 100)}%`

interface Props {
  curves: Record<string, number[]>
  /** Yous with nothing to go on: drawn like the others, noted in the table. */
  abstained: Set<string>
}

/** Rolling accuracy (over the last 10 picks) per pick, one line per You. */
export default function LearningCurve({ curves, abstained }: Props) {
  const ids = YOUS.map((y) => y.id).filter((id) => curves[id]?.length)
  const n = Math.max(0, ...ids.map((id) => curves[id].length))
  const [hover, setHover] = useState<number | null>(null)
  if (n === 0) return null

  const values = ids.flatMap((id) => curves[id])
  let lo = Math.min(Math.floor(Math.min(...values) * 10) / 10, 0.4)
  let hi = Math.max(Math.ceil(Math.max(...values) * 10) / 10, 0.6)
  lo = Math.max(0, lo)
  hi = Math.min(1, hi)
  const step = hi - lo > 0.5 ? 0.2 : 0.1
  const ticks: number[] = []
  for (let t = Math.ceil(lo / step - 1e-9) * step; t <= hi + 1e-9; t += step) ticks.push(Math.round(t * 100) / 100)

  const x = (k: number) => M.left + (n === 1 ? 0 : ((k - 1) / (n - 1)) * (W - M.left - M.right))
  const y = (v: number) => M.top + (1 - (v - lo) / (hi - lo)) * (H - M.top - M.bottom)

  // Direct labels at the line ends, nudged apart so they never overlap.
  const ends = ids
    .map((id) => ({ id, y: y(curves[id][curves[id].length - 1]) }))
    .sort((a, b) => a.y - b.y)
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + LABEL_GAP)
  const over = ends.length ? ends[ends.length - 1].y - (H - M.bottom) : 0
  if (over > 0) ends.forEach((e) => (e.y -= over))

  const at = (e: PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const px = ((e.clientX - r.left) / r.width) * W
    const k = Math.round(1 + ((px - M.left) / (W - M.left - M.right)) * (n - 1))
    setHover(Math.min(n, Math.max(1, k)))
  }

  const checkpoints = [...new Set([5, 10, 20, 30, n].filter((k) => k <= n))].sort((a, b) => a - b)
  const tipLeft = hover ? Math.min(78, Math.max(22, (x(hover) / W) * 100)) : 0

  return (
    <figure className="curve">
      <ul className="legend" aria-label="Legend">
        {ids.map((id) => (
          <li key={id} style={{ '--c': youById(id).color } as CSSProperties}>
            <span className="key" />
            {youById(id).name}
          </li>
        ))}
      </ul>
      <div className="curve-box">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label={`Rolling accuracy over the last 10 picks, for ${ids.map((id) => youById(id).name).join(', ')}. A table follows.`}
          onPointerMove={at}
          onPointerDown={at}
          onPointerLeave={() => setHover(null)}
        >
          {ticks.map((t) => (
            <g key={t}>
              <line className="grid" x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} />
              <text className="tick" x={M.left - 8} y={y(t)} textAnchor="end" dominantBaseline="middle">
                {pct(t)}
              </text>
            </g>
          ))}
          <line className="coin" x1={M.left} x2={W - M.right} y1={y(0.5)} y2={y(0.5)} />
          <text className="tick" x={M.left} y={H - 8} textAnchor="start">
            pick 1
          </text>
          <text className="tick" x={W - M.right} y={H - 8} textAnchor="end">
            pick {n}
          </text>
          {ids.map((id) => (
            <polyline
              key={id}
              className="line"
              style={{ '--c': youById(id).color } as CSSProperties}
              points={curves[id].map((v, i) => `${x(i + 1).toFixed(1)},${y(v).toFixed(1)}`).join(' ')}
            />
          ))}
          {ends.map((e) => (
            <text
              key={e.id}
              className="end"
              x={W - M.right + 8}
              y={e.y}
              dominantBaseline="middle"
            >
              {youById(e.id).name.replace(' you', '')}
            </text>
          ))}
          {hover !== null && (
            <g>
              <line className="cross" x1={x(hover)} x2={x(hover)} y1={M.top} y2={H - M.bottom} />
              {ids.map(
                (id) =>
                  curves[id][hover - 1] !== undefined && (
                    <circle
                      key={id}
                      className="dot"
                      style={{ '--c': youById(id).color } as CSSProperties}
                      cx={x(hover)}
                      cy={y(curves[id][hover - 1])}
                      r={5}
                    />
                  ),
              )}
            </g>
          )}
        </svg>
        {hover !== null && (
          <div className="tip" style={{ left: `${tipLeft}%` }} role="status">
            <div className="tip-head">pick {hover}</div>
            {ids.map((id) => (
              <div className="tip-row" key={id} style={{ '--c': youById(id).color } as CSSProperties}>
                <span className="key" />
                <strong>{curves[id][hover - 1] === undefined ? '–' : pct(curves[id][hover - 1])}</strong>
                <span>{youById(id).name.replace(' you', '')}</span>
              </div>
            ))}
          </div>
        )}
      </div>
      <figcaption>
        Each line is the share of the last 10 picks that You called right. Revealed starts at a coin
        flip because it has nothing to learn from yet; the other three were fixed before your first
        clothing pick.
      </figcaption>
      <details className="as-table">
        <summary>Show as a table</summary>
        <table>
          <thead>
            <tr>
              <th scope="col">Accuracy over the last 10 picks</th>
              {checkpoints.map((k) => (
                <th scope="col" key={k}>
                  pick {k}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {ids.map((id) => (
              <tr key={id}>
                <th scope="row">
                  {youById(id).name}
                  {abstained.has(id) && <small> (abstained)</small>}
                </th>
                {checkpoints.map((k) => (
                  <td key={k}>{curves[id][k - 1] === undefined ? '–' : pct(curves[id][k - 1])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </figure>
  )
}
