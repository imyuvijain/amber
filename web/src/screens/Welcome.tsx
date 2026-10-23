import { useState, type CSSProperties, type FormEvent } from 'react'
import { ApiError, createUser } from '../api'
import { YOUS } from '../yous'
import './Welcome.css'

interface Props {
  onCreated: (userId: string) => void
}

export default function Welcome({ onCreated }: Props) {
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function submit(e: FormEvent) {
    e.preventDefault()
    if (busy) return
    setBusy(true)
    setError(null)
    try {
      const { user_id } = await createUser(name.trim() || 'Anonymous')
      onCreated(user_id)
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Something went wrong. Try again.')
      setBusy(false)
    }
  }

  return (
    <section className="welcome">
      <h1>
        Four of you
        <br />
        are about to bet
        <br />
        on your next pick.
      </h1>
      <p className="lede">
        Each one is a different guess at who you are. You answer a few questions, choose between
        artworks, and then make 40 clothing picks. Every one of those picks is predicted before
        you make it. The scoreboard shows which one knew you best.
      </p>

      <ul className="lineup" aria-label="The four Yous">
        {YOUS.map((y, i) => (
          <li key={y.id} style={{ '--c': y.color, '--i': i } as CSSProperties}>
            <span className="bead" />
            <span className="lineup-name">{y.name}</span>
            <span className="lineup-pitch">{y.pitch}</span>
          </li>
        ))}
      </ul>

      <form className="start" onSubmit={submit}>
        <label htmlFor="name">What should we call you?</label>
        <div className="start-row">
          <input
            id="name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="A first name is fine"
            autoComplete="given-name"
            maxLength={60}
          />
          <button className="btn" type="submit" disabled={busy}>
            {busy ? 'Starting…' : 'Start'}
          </button>
        </div>
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
      </form>
    </section>
  )
}
