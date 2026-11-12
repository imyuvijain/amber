import { useEffect, useState, type CSSProperties } from 'react'
import { ApiError, getResults, type Results as ResultsData, type Score } from '../api'
import '../components/Form.css'
import LearningCurve from '../components/LearningCurve'
import PairScreen from '../components/PairScreen'
import RetestCard from '../components/RetestCard'
import Scoreboard from '../components/Scoreboard'
import YouMap from '../components/YouMap'
import { YOUS, youById } from '../yous'
import './Results.css'
import type { ScreenProps } from './props'

/** The headline You: the art you, or (when art abstained) the best non-revealed You with a say. */
function headliner(results: ResultsData): Score | undefined {
  const speaks = (s: Score) => s.you in results.cosines
  const art = results.scores_all.find((s) => s.you === 'art')
  if (art && speaks(art)) return art
  return results.scores_all
    .filter((s) => s.you !== 'revealed' && speaks(s))
    .sort((a, b) => b.accuracy - a.accuracy)[0]
}

/** Only a score whose 95% range clears 50% is a win; one that sits wholly below is worse than
 *  guessing; anything else is indistinguishable from a coin flip. */
function verdictOf(s: Score): 'win' | 'chance' | 'worse' {
  if (s.ci_low > 0.5) return 'win'
  if (s.ci_high < 0.5) return 'worse'
  return 'chance'
}

function Headline({ results }: { results: ResultsData }) {
  const lead = headliner(results)
  const n = results.n_clothes
  if (!lead) {
    return (
      <header className="headline">
        <h1>Nothing to predict from.</h1>
        <p className="lede">None of the frozen Yous had anything to go on, so only Revealed is left to read.</p>
      </header>
    )
  }
  const you = youById(lead.you)
  const subject = lead.you === 'art' ? 'Your art taste' : you.name
  const right = Math.round(lead.correct)
  const verdict = verdictOf(lead)
  const others = results.scores_all.filter((s) => s.you !== lead.you)
  return (
    <header className="headline">
      <h1>
        {subject} {verdict === 'win' ? 'predicted' : 'called'}{' '}
        <span style={{ '--c': you.color } as CSSProperties} className="mark" data-verdict={verdict}>
          {right}
        </span>{' '}
        of your {lead.n} picks
        {verdict === 'chance' && ', about what a coin flip gets'}
        {verdict === 'worse' && ', worse than a coin flip'}
      </h1>
      <p className="ci">
        95% range {Math.round(lead.ci_low * lead.n)}–{Math.round(lead.ci_high * lead.n)} of {lead.n}
        {lead.n !== n ? ` (out of ${n} picks)` : ''}
      </p>
      <ul className="compare" aria-label="For comparison">
        <li>
          <span className="who">Coin flip</span>
          <strong>{Math.round(lead.n / 2)}</strong>
        </li>
        {others.map((s) => (
          <li key={s.you} style={{ '--c': youById(s.you).color } as CSSProperties}>
            <span className="who">
              <span className="bead" />
              {youById(s.you).name}
            </span>
            <strong>{s.you in results.cosines ? Math.round(s.correct) : 'abstained'}</strong>
          </li>
        ))}
      </ul>
    </header>
  )
}

function ReadoutBlock({ results }: { results: ResultsData }) {
  const { sentences } = results.readout
  return (
    <section className="block" aria-labelledby="readout-h">
      <h2 id="readout-h">What it says</h2>
      {sentences.length ? (
        <ul className="sentences">
          {sentences.map((s) => (
            <li key={s}>{s}</li>
          ))}
        </ul>
      ) : (
        <p className="lede">Not enough to read yet.</p>
      )}
    </section>
  )
}

function Cosines({ cosines }: { cosines: ResultsData['cosines'] }) {
  const ids = YOUS.map((y) => y.id).filter((id) => id in cosines)
  const missing = YOUS.filter((y) => !(y.id in cosines))
  if (ids.length < 2) return null
  return (
    <section className="block" aria-labelledby="cos-h">
      <h2 id="cos-h">How close the Yous are</h2>
      <p className="lede">
        Cosine between each pair of directions: 1 means the same taste, 0 unrelated, negative
        opposed.
      </p>
      <div className="scroll-x">
      <table className="cosines">
        <thead>
          <tr>
            <td />
            {ids.map((id) => (
              <th scope="col" key={id} style={{ '--c': youById(id).color } as CSSProperties}>
                <span className="bead" />
                {youById(id).name.replace(' you', '')}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {ids.map((a) => (
            <tr key={a}>
              <th scope="row" style={{ '--c': youById(a).color } as CSSProperties}>
                <span className="bead" />
                {youById(a).name.replace(' you', '')}
              </th>
              {ids.map((b) => {
                const v = cosines[a][b]
                return (
                  <td
                    key={b}
                    className={a === b ? 'self' : undefined}
                    style={{ '--t': `${Math.round(Math.min(1, Math.abs(v)) * 45)}%` } as CSSProperties}
                  >
                    {a === b ? '–' : v.toFixed(2)}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
      </div>
      {missing.length > 0 && (
        <p className="note">Left out, nothing to go on: {missing.map((y) => y.name).join(', ')}.</p>
      )}
    </section>
  )
}

export default function Results({ userId, reset }: ScreenProps) {
  const [results, setResults] = useState<ResultsData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [retesting, setRetesting] = useState(false)

  useEffect(() => {
    if (retesting) return
    let live = true
    getResults(userId)
      .then((r) => live && setResults(r))
      .catch((err) => live && setError(err instanceof ApiError ? err.message : 'Could not load results.'))
    return () => {
      live = false
    }
  }, [userId, retesting])

  if (retesting) {
    return (
      <PairScreen
        userId={userId}
        domain="clothes"
        phase="retest"
        heading="Same pairs, a week later"
        hint="Pick again, the way you would today. These picks change none of your scores; they measure how steady your taste is."
        progressLabel="Retest picks"
        dwellMs={300}
        onDone={() => {
          setResults(null) // reload with the ceiling, not the old banner
          setRetesting(false)
        }}
        onReset={reset}
      />
    )
  }
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

  const abstained = new Set(YOUS.map((y) => y.id).filter((id) => !(id in results.cosines)))
  return (
    <div className="results">
      <Headline results={results} />
      <RetestCard results={results} onStart={() => setRetesting(true)} />
      <section className="block" aria-labelledby="score-h">
        <h2 id="score-h">Scoreboard</h2>
        <Scoreboard results={results} />
      </section>
      <section className="block" aria-labelledby="curve-h">
        <h2 id="curve-h">Learning curve</h2>
        <LearningCurve curves={results.curves} abstained={abstained} />
      </section>
      <ReadoutBlock results={results} />
      <section className="block" aria-labelledby="map-h">
        <h2 id="map-h">Map of your Yous</h2>
        <YouMap data={results.map} />
      </section>
      <Cosines cosines={results.cosines} />
      <div className="form-actions">
        <button className="btn btn-quiet" type="button" onClick={reset}>
          Play again
        </button>
      </div>
    </div>
  )
}
