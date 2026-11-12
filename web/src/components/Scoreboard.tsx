import type { CSSProperties } from 'react'
import type { Results, Score } from '../api'
import { YOUS, youById } from '../yous'
import './Scoreboard.css'

const pct = (x: number) => `${(x * 100).toFixed(0)}%`
const place = (x: number) => `${(Math.min(1, Math.max(0, x)) * 100).toFixed(2)}%`

function order(scores: Score[]): Score[] {
  const rank = (id: string) => {
    const i = YOUS.findIndex((y) => y.id === id)
    return i < 0 ? YOUS.length : i
  }
  return [...scores].sort((a, b) => rank(a.you) - rank(b.you))
}

function Row({ score, late, abstained }: { score: Score; late?: Score; abstained: boolean }) {
  const you = youById(score.you)
  return (
    <li className="row" style={{ '--c': you.color } as CSSProperties}>
      <span className="row-name">
        <span className="bead" />
        <span>
          {you.name}
          {abstained && <small className="row-abstain">abstained</small>}
        </span>
      </span>
      <span
        className="row-plot"
        role="img"
        aria-label={
          abstained
            ? `${you.name}: abstained, scored as a coin flip`
            : `${you.name}: ${pct(score.accuracy)} right, 95% interval ${pct(score.ci_low)} to ${pct(score.ci_high)}`
        }
      >
        <span className="bar" style={{ width: place(score.accuracy) }} />
        {!abstained && (
          <span
            className="whisker"
            style={{ left: place(score.ci_low), width: `calc(${place(score.ci_high)} - ${place(score.ci_low)})` }}
          />
        )}
      </span>
      <span className="row-num">
        {pct(score.accuracy)}
        <small className="row-late">{late ? pct(late.accuracy) : '–'}</small>
      </span>
    </li>
  )
}

/** One row per You: accuracy bar, 95% whisker, dashed coin-flip line. */
export default function Scoreboard({ results }: { results: Results }) {
  const lateN = results.scores_late[0]?.n ?? 0
  const lateLabel = lateN > 0 ? `picks ${results.n_clothes - lateN + 1}–${results.n_clothes}` : 'late picks'
  return (
    <div className="scoreboard">
      <p className="lede">
        Each You predicted all {results.n_clothes} clothing pairs before you answered. Bars show how
        often it was right across all of them; the whisker is the 95% range that sampling luck alone
        could explain. The small number is the same score over {lateLabel}, once Revealed had time
        to learn from you.
      </p>
      <div className="chart">
        <div className="chart-head">
          <span className="coin">coin flip</span>
          <span className="head-num">
            <span>all {results.n_clothes}</span>
            <span>{lateLabel}</span>
          </span>
        </div>
        <ul className="rows">
          {order(results.scores_all).map((s) => (
            <Row
              key={s.you}
              score={s}
              late={results.scores_late.find((l) => l.you === s.you)}
              abstained={!(s.you in results.cosines)}
            />
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
    </div>
  )
}
