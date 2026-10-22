import type { Item } from '../api'
import type { Status } from '../hooks/usePairFlow'
import './PairBoard.css'

interface Props {
  a: Item
  b: Item
  status: Status
  picked: string | null
  onPick: (item: Item) => void
}

const SIDES = ['Left', 'Right'] as const

/** Two pictures, nothing else: titles stay hidden so the pick is about looks alone. */
export default function PairBoard({ a, b, status, picked, onPick }: Props) {
  const locked = status !== 'ready'
  return (
    <div className="board" role="group" aria-label="Choose one">
      {[a, b].map((item, i) => (
        <button
          key={item.id}
          type="button"
          className="tile"
          disabled={locked}
          data-state={picked === null ? 'open' : picked === item.id ? 'picked' : 'passed'}
          aria-label={`${SIDES[i]} option`}
          aria-keyshortcuts={i === 0 ? 'ArrowLeft' : 'ArrowRight'}
          onClick={() => onPick(item)}
        >
          <img src={item.thumb_url} alt="" draggable={false} />
        </button>
      ))}
      <span className="board-or" aria-hidden="true">
        or
      </span>
    </div>
  )
}

export function ProgressBar({ done, total, label }: { done: number; total: number; label: string }) {
  const pct = total > 0 ? Math.min(100, (done / total) * 100) : 0
  return (
    <div className="progress">
      <div
        className="progress-track"
        role="progressbar"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={total}
        aria-valuenow={done}
      >
        <div className="progress-fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="progress-count">
        {done} of {total}
      </span>
    </div>
  )
}
