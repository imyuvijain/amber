import { useMemo, useState, type CSSProperties, type PointerEvent } from 'react'
import { thumbUrl, type MapPoint, type YouMapData } from '../api'
import { youById } from '../yous'
import './YouMap.css'

const W = 640
const H = 460
const PAD = 52
const HIT = 18 // pointer reach, in svg units, of the nearest-dot hover
const CHIP_W = 150

interface Props {
  data: YouMapData
}

/** The clothes 2-D plane: clothing and art as dots, each You as a labeled marker at the
 *  edge of the cloud, in the direction its taste points. */
export default function YouMap({ data }: Props) {
  const [near, setNear] = useState<MapPoint | null>(null)
  const [picked, setPicked] = useState<string | null>(null)

  const view = useMemo(() => {
    const all = [...data.points, ...data.yous, { x: 0, y: 0 }]
    const xs = all.map((p) => p.x)
    const ys = all.map((p) => p.y)
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)]
    const k = Math.min((W - 2 * PAD) / (x1 - x0 || 1), (H - 2 * PAD) / (y1 - y0 || 1))
    const cx = (x0 + x1) / 2
    const cy = (y0 + y1) / 2
    const sx = (v: number) => W / 2 + (v - cx) * k
    const sy = (v: number) => H / 2 - (v - cy) * k
    const dots = data.points.map((p) => ({ p, sx: sx(p.x), sy: sy(p.y) }))
    // Marker labels sit above their marker (below near the top edge); a label that would
    // cover an earlier one flips to the other side.
    const placed: { x: number; y: number }[] = []
    const markers = data.yous.map((m) => {
      const mx = sx(m.x)
      const my = sy(m.y)
      const lx = Math.min(W - CHIP_W / 2, Math.max(CHIP_W / 2, mx))
      let ly = my < 56 ? my + 32 : my - 22
      if (placed.some((q) => Math.abs(q.x - lx) < CHIP_W && Math.abs(q.y - ly) < 26)) {
        ly = ly < my ? my + 32 : my - 22
      }
      placed.push({ x: lx, y: ly })
      return { m, mx, my, lx, ly }
    })
    return { dots, markers, ox: sx(0), oy: sy(0) }
  }, [data])

  if (!data.points.length) return null

  const track = (e: PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    const px = ((e.clientX - r.left) / r.width) * W
    const py = ((e.clientY - r.top) / r.height) * H
    let best: MapPoint | null = null
    let bestD = HIT * HIT
    for (const d of view.dots) {
      const dd = (d.sx - px) ** 2 + (d.sy - py) ** 2
      if (dd < bestD) {
        best = d.p
        bestD = dd
      }
    }
    setNear(best)
  }
  const hovered = near ? view.dots.find((d) => d.p.id === near.id) : null
  const chosen = data.yous.find((m) => m.you === picked)
  const toggle = (id: string) => setPicked(picked === id ? null : id)

  return (
    <figure className="map">
      <ul className="legend map-legend" aria-label="Legend">
        <li>
          <svg width="12" height="12" aria-hidden="true">
            <circle cx="6" cy="6" r="4.5" className="dot-clothes" />
          </svg>
          a clothing item
        </li>
        <li>
          <svg width="12" height="12" aria-hidden="true">
            <rect x="1.5" y="1.5" width="9" height="9" rx="1.5" className="dot-art" />
          </svg>
          a painting
        </li>
        <li>
          <span className="bead" style={{ '--c': 'var(--ink)', '--size': '0.8rem' } as CSSProperties} />a You
        </li>
      </ul>
      <div className="map-box">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          role="group"
          aria-label={`Map of ${data.points.length} items and ${data.yous.length} Yous. Each You points toward the kind of item it likes. The closeness table below gives the same thing as numbers.`}
          onPointerMove={track}
          onPointerDown={track}
          onPointerLeave={() => setNear(null)}
        >
          {view.dots.map(({ p, sx, sy }) =>
            p.domain === 'art' ? (
              <rect key={p.id} className="dot-art" x={sx - 3.5} y={sy - 3.5} width={7} height={7} rx={1.5} />
            ) : (
              <circle key={p.id} className="dot-clothes" cx={sx} cy={sy} r={3.5} />
            ),
          )}
          {hovered && <circle className="ring" cx={hovered.sx} cy={hovered.sy} r={7} />}
          {view.markers.map(({ m, mx, my, lx, ly }) => {
            const you = youById(m.you)
            const on = picked === m.you
            return (
              <g
                key={m.you}
                className="you-marker"
                data-on={on}
                style={{ '--c': you.color } as CSSProperties}
                tabIndex={0}
                role="button"
                aria-pressed={on}
                aria-label={`${you.name}: show what it likes most`}
                onClick={() => toggle(m.you)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault()
                    toggle(m.you)
                  }
                }}
              >
                <line className="stem" x1={view.ox} y1={view.oy} x2={mx} y2={my} />
                <circle className="hit" cx={mx} cy={my} r={22} />
                <circle className="marker" cx={mx} cy={my} r={11} />
                <text className="marker-label" x={lx} y={ly} textAnchor="middle">
                  {you.name}
                </text>
              </g>
            )
          })}
        </svg>
        {hovered && (
          <img
            className="thumb-tip"
            src={thumbUrl(hovered.p.id)}
            alt=""
            style={{
              left: `${Math.min(86, Math.max(14, (hovered.sx / W) * 100))}%`,
              top: `${(hovered.sy / H) * 100}%`,
            }}
          />
        )}
      </div>
      <figcaption>
        A flat picture of the clothes space, fitted to the clothes you could see. Each marker sits at
        the edge and points the way its You leans; only the direction counts. The dots are{' '}
        {data.points.filter((p) => p.domain === 'clothes').length} of those clothes and{' '}
        {data.points.filter((p) => p.domain === 'art').length} paintings. Hover a dot for its
        picture.
        {chosen ? '' : ' Tap a You to see the items it likes most.'}
      </figcaption>
      {chosen && (
        <div className="top-strip" style={{ '--c': youById(chosen.you).color } as CSSProperties}>
          <p>
            <span className="bead" /> {youById(chosen.you).name} likes these most
          </p>
          <ul>
            {chosen.top.slice(0, 6).map((id) => (
              <li key={id}>
                <img src={thumbUrl(id)} alt="" loading="lazy" />
              </li>
            ))}
          </ul>
        </div>
      )}
    </figure>
  )
}
