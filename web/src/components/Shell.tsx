import type { ReactNode } from 'react'
import { STEPS } from '../steps'
import './Shell.css'

interface Props {
  step: string
  signedIn: boolean
  onReset: () => void
  children: ReactNode
}

/** Page frame: wordmark, the step trail, and the current screen. */
export default function Shell({ step, signedIn, onReset, children }: Props) {
  const trail = STEPS.filter((s) => s.id !== 'welcome')
  const current = trail.findIndex((s) => s.id === step)
  return (
    <div className="shell">
      <header className="shell-bar">
        <span className="wordmark">
          <svg viewBox="0 0 32 32" aria-hidden="true" width="22" height="22">
            <path
              d="M16 3c5 6.5 9 11 9 16a9 9 0 0 1-18 0c0-5 4-9.5 9-16z"
              fill="var(--amber)"
            />
            <ellipse cx="12.5" cy="19" rx="2" ry="3.5" fill="#fff" opacity=".45" transform="rotate(18 12.5 19)" />
          </svg>
          amber
        </span>
        {step !== 'welcome' && (
          <nav aria-label="Progress">
            <ol className="trail">
              {trail.map((s, i) => (
                <li
                  key={s.id}
                  aria-current={i === current ? 'step' : undefined}
                  data-state={i < current ? 'past' : i === current ? 'now' : 'next'}
                >
                  {s.label}
                </li>
              ))}
            </ol>
          </nav>
        )}
        {signedIn && step !== 'welcome' && (
          <button type="button" className="link" onClick={onReset}>
            Start over
          </button>
        )}
      </header>
      <main className="shell-main" key={step}>
        {children}
      </main>
    </div>
  )
}
