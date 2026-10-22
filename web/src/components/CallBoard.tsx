import type { CSSProperties } from 'react'
import type { Reveal } from '../api'
import type { Tally } from '../hooks/usePairFlow'
import { YOUS } from '../yous'
import './CallBoard.css'

interface Props {
  tally: Tally
  /** The calls on the pair just answered; null between picks. */
  reveal: Reveal | null
}

/** Which Yous called the last pick, with a running tally. Abstaining Yous never appear. */
export default function CallBoard({ tally, reveal }: Props) {
  const shown = YOUS.filter((y) => y.id in tally || (reveal !== null && y.id in reveal))
  return (
    <div className="calls" aria-live="polite">
      {shown.length === 0 ? (
        <p className="calls-empty">Your Yous have already locked in their guesses. Pick one.</p>
      ) : (
        <ul>
          {shown.map((y) => {
            const call = reveal?.[y.id]
            const row = tally[y.id] ?? { hit: 0, n: 0 }
            return (
              <li
                key={y.id}
                style={{ '--c': y.color } as CSSProperties}
                data-call={call === undefined ? 'none' : call ? 'hit' : 'miss'}
              >
                <span className="bead" />
                <span className="calls-name">{y.name}</span>
                <span className="calls-verdict">
                  {call === undefined ? '' : call ? 'Called it' : 'Missed'}
                </span>
                <span className="calls-tally">
                  {row.hit}/{row.n}
                </span>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
