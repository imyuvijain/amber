import { useEffect, type ReactNode } from 'react'
import type { Domain, Phase } from '../api'
import { usePairFlow } from '../hooks/usePairFlow'
import CallBoard from './CallBoard'
import PairBoard, { ProgressBar } from './PairBoard'
import './Form.css'
import './PairScreen.css'

interface Props {
  userId: string
  domain: Domain
  phase: Phase
  heading: string
  hint: ReactNode
  progressLabel: string
  /** Show which Yous called each pick, plus the running tally. */
  showCalls?: boolean
  dwellMs: number
  onDone: () => void
  onReset: () => void
}

/** Shared by ThisOrThat and Showdown: heading, the pair, keys, progress, optional calls. */
export default function PairScreen(props: Props) {
  const { userId, domain, phase, heading, hint, progressLabel, showCalls, dwellMs } = props
  const flow = usePairFlow({ userId, domain, phase, dwellMs, onDone: props.onDone })
  const { pair, pick, status } = flow

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.repeat || e.metaKey || e.ctrlKey || e.altKey || !pair) return
      if (e.key === 'ArrowLeft') {
        e.preventDefault()
        void pick(pair.a)
      } else if (e.key === 'ArrowRight') {
        e.preventDefault()
        void pick(pair.b)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [pair, pick])

  const { failure } = flow
  return (
    <section className="pair-screen">
      <header>
        <h2>{heading}</h2>
        <p className="lede">{hint}</p>
      </header>

      {pair ? (
        <PairBoard a={pair.a} b={pair.b} status={status} picked={flow.picked} onPick={pick} />
      ) : (
        <div className="board-empty" aria-busy={failure === null}>
          {failure === null && 'Loading…'}
        </div>
      )}

      {flow.progress && (
        <ProgressBar done={flow.progress.done} total={flow.progress.total} label={progressLabel} />
      )}
      <p className="keys" aria-hidden="true">
        <kbd>←</kbd> <kbd>→</kbd> to choose
      </p>

      {showCalls && <CallBoard tally={flow.tally} reveal={flow.reveal} />}

      {failure && (
        <div className="pair-error" role="alert">
          <p className="error">{failure.message}</p>
          {failure.kind === 'load' && (
            <div className="form-actions">
              <button className="btn btn-quiet" type="button" onClick={() => void flow.retry()}>
                Try again
              </button>
              {failure.status === 404 && (
                <button className="link" type="button" onClick={props.onReset}>
                  Start over
                </button>
              )}
            </div>
          )}
        </div>
      )}
    </section>
  )
}
