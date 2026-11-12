import type { Results } from '../api'
import './RetestCard.css'

const day = (d: Date) => d.toLocaleDateString(undefined, { weekday: 'long', day: 'numeric', month: 'long' })
const clock = (d: Date) => d.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' })

interface Props {
  results: Results
  onStart: () => void
}

/** The retest a week later: a date to come back on, then a button, then (on the scoreboard) the
 *  ceiling. Renders nothing before the clothes picks are done or once the retest is finished. */
export default function RetestCard({ results, onStart }: Props) {
  if (results.retest_due_at === null || results.ceiling !== null) return null
  const n = results.retest_pairs
  if (results.retest_due) {
    return (
      <aside className="retest" aria-labelledby="retest-h">
        <h2 id="retest-h">Your {n}-pick retest is ready</h2>
        <p className="lede">
          The same pairs again, some flipped left to right. How often you pick the same one is the
          most any You could be expected to get right.
        </p>
        <div className="form-actions">
          <button className="btn" type="button" onClick={onStart}>
            Start the retest
          </button>
        </div>
      </aside>
    )
  }
  const opens = new Date(results.retest_due_at)
  return (
    <aside className="retest" aria-labelledby="retest-h">
      <h2 id="retest-h">
        Come back on {day(opens)} for a {n}-pick retest
      </h2>
      <p className="lede">
        We will show you {n} of these pairs again. Opens at {clock(opens)}. Nobody can predict your
        picks better than you can repeat them, so this puts a ceiling on every score here.
      </p>
    </aside>
  )
}
