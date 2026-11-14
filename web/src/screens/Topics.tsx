import { useCallback, useEffect, useState } from 'react'
import { ApiError, getTopics, tapTopic, type Topic, type TopicAction, type TopicList } from '../api'
import './Topics.css'

const SHOWN = 20

/** The veto screen: topics the engine thinks matter to you now. Tap ✕ on wrong ones. */
export default function Topics() {
  const [list, setList] = useState<TopicList | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [undo, setUndo] = useState<Topic | null>(null)

  useEffect(() => {
    getTopics()
      .then(setList)
      .catch((e: unknown) => setError(e instanceof ApiError ? e.message : String(e)))
  }, [])

  const tap = useCallback(
    async (t: Topic, action: TopicAction) => {
      if (!list || busy) return
      setBusy(t.id)
      try {
        const shown = list.topics.slice(0, SHOWN).map((x) => x.id)
        const next = await tapTopic(t.id, action, shown)
        setList(next)
        setUndo(action === 'veto' ? t : null)
        setError(null)
      } catch (e) {
        setError(e instanceof ApiError ? e.message : String(e))
      } finally {
        setBusy(null)
      }
    },
    [list, busy],
  )

  if (error && !list) return <p className="topics-error" role="alert">{error}</p>
  if (!list) return <p className="topics-loading">Loading your topics…</p>

  const shown = list.topics.slice(0, SHOWN)
  return (
    <section className="topics" aria-labelledby="topics-title">
      <header className="topics-head">
        <h1 id="topics-title">What matters to you right now</h1>
        <p>
          Ranked from your own activity. Tap <span aria-hidden="true">✕</span>
          <span className="sr-only">the cross</span> on anything that isn’t you,{' '}
          <span aria-hidden="true">↑</span>
          <span className="sr-only">the arrow</span> on what matters more.
        </p>
        <p className="topics-learned">
          {list.n_labels === 0 ? 'No taps yet: this is the frozen score.' : `Tuned by ${list.n_labels} labels.`}
        </p>
      </header>

      {error && <p className="topics-error" role="alert">{error}</p>}

      <ol className="topic-list">
        {shown.map((t, i) => (
          <li key={t.id} className="topic" data-state={t.state ?? undefined} aria-busy={busy === t.id}>
            <span className="topic-rank" aria-hidden="true">{i + 1}</span>
            <div className="topic-body">
              <div className="topic-name">
                {t.name}
                {t.state === 'boost' && <span className="topic-tag">boosted</span>}
              </div>
              <div
                className="topic-meter"
                role="meter"
                aria-label="relevance"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(t.relevance * 100)}
              >
                <span style={{ width: `${Math.max(2, t.relevance * 100)}%` }} />
              </div>
              <ul className="topic-drivers">
                {t.drivers.slice(0, 3).map((d) => (
                  <li key={d}>{d}</li>
                ))}
              </ul>
            </div>
            <div className="topic-actions">
              <button
                type="button"
                className="tap tap-boost"
                aria-pressed={t.state === 'boost'}
                aria-label={t.state === 'boost' ? `Remove boost from ${t.name}` : `Boost ${t.name}`}
                disabled={busy !== null}
                onClick={() => tap(t, t.state === 'boost' ? 'clear' : 'boost')}
              >
                ↑
              </button>
              <button
                type="button"
                className="tap tap-veto"
                aria-label={`Not me: ${t.name}`}
                disabled={busy !== null}
                onClick={() => tap(t, 'veto')}
              >
                ✕
              </button>
            </div>
          </li>
        ))}
      </ol>

      {list.muted.length > 0 && (
        <details className="topics-muted">
          <summary>Muted ({list.muted.length})</summary>
          <ul>
            {list.muted.map((t) => (
              <li key={t.id}>
                <span>{t.name}</span>
                <button type="button" className="link" disabled={busy !== null} onClick={() => tap(t, 'clear')}>
                  Undo
                </button>
              </li>
            ))}
          </ul>
        </details>
      )}

      {undo && (
        <div className="topics-toast" role="status">
          Muted <strong>{undo.name}</strong>
          <button type="button" className="link" onClick={() => tap(undo, 'clear')}>
            Undo
          </button>
        </div>
      )}
    </section>
  )
}
