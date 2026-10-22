import { useEffect, useState, type CSSProperties } from 'react'
import { ApiError, getResults, type Results, type Score } from '../api'
import '../components/Form.css'
import { YOUS, youById } from '../yous'
import './Scoreboard.css'
import type { ScreenProps } from './props'

const pct = (x: number) => `${(x * 100).toFixed(0)}%`
const place = (x: number) => `${(Math.min(1, Math.max(0, x)) * 100).toFixed(2)}%`

function order(scores: Score[]): Score[] {
  const rank = (id: string) => {
    const i = YOUS.findIndex((y) => y.id === id)
    return i < 0 ? YOUS.length : i
  }
  return [...scores].sort((a, b) => rank(a.you) - rank(b.you))
}

function headline(scores: Score[]): string {
  const ranked = [...scores].sort((a, b) => b.accuracy - a.accuracy)
  const [top, second] = ranked
  if (!top) return 'No showdown picks yet.'
  const name = youById(top.you).name
  if (!second || second.accuracy === top.accuracy) return `${name} is tied for the lead.`
  return top.ci_low > second.ci_high
    ? `${name} knew you best.`
    : `${name} leads, but it is close.`
}

function Row({ score }: { score: Score }) {
  const you = youById(score.you)
  return (
    <li className="row" style={{ '--c': you.color } as CSSProperties}>
      <span className="row-name">
        <span className="bead" />
        {you.name}
      </span>
      <span
        className="row-plot"
        role="img"
        aria-label={`${you.name}: ${pct(score.accuracy)} right, 95% interval ${pct(score.ci_low)} to ${pct(score.ci_high)}`}
      >
        <span className="bar" style={{ width: place(score.accuracy) }} />
        <span
          className="whisker"
          style={{ left: place(score.ci_low), width: `calc(${place(score.ci_high)} - ${place(score.ci_low)})` }}
        />
      </span>
      <span className="row-num">{pct(score.accuracy)}</span>
    </li>
  )
}

export default function Scoreboard({ userId, reset }: ScreenProps) {
  const [results, setResults] = useState<Results | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let live = true
    getResults(userId)
      .then((r) => live && setResults(r))
      .catch((err) => live && setError(err instanceof ApiError ? err.message : 'Could not load results.'))
    return () => {
      live = false
    }
  }, [userId])

  if (error) {
    return (
      <div className="form-screen">
        <p className="error" role="alert">
          {error}
        </p>
        <div className="form-actions">
          <button className="btn btn-quiet" type="button" onClick={reset}>
            Start over
          </button>
        </div>
      </div>
    )
  }
  if (!results) return <p className="lede">Counting…</p>

  return (
    <section className="scoreboard">
      <h1>{headline(results.scores)}</h1>
      <p className="lede">
        Each You predicted all {results.n_showdown} showdown pairs before you answered. Bars show
        how often it was right; the whisker is the 95% range that sampling luck alone could explain.
      </p>

      <div className="chart">
        <div className="chart-head">
          <span className="coin">coin flip</span>
        </div>
        <ul className="rows">
          {order(results.scores).map((s) => (
            <Row key={s.you} score={s} />
          ))}
        </ul>
        <div className="axis" aria-hidden="true">
          <span>0%</span>
          <span>100%</span>
        </div>
      </div>

      <p className="note">
        A You with nothing to go on abstains and scores 50% on every pair. Yous that sit on the
        dashed line did no better than guessing.
      </p>
      <div className="form-actions">
        <button className="btn btn-quiet" type="button" onClick={reset}>
          Play again
        </button>
      </div>
    </section>
  )
}
