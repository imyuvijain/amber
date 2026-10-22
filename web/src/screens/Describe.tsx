import { useState, type FormEvent } from 'react'
import { ApiError, setDescribe } from '../api'
import '../components/Form.css'
import './Describe.css'
import type { ScreenProps } from './props'

export default function Describe({ userId, advance }: ScreenProps) {
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function submit(e: FormEvent) {
    e.preventDefault()
    if (busy || !text.trim()) return
    setBusy(true)
    setError(null)
    try {
      await setDescribe(userId, text.trim())
      advance()
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not save that. Try again.')
      setBusy(false)
    }
  }

  return (
    <form className="form-screen" onSubmit={submit}>
      <h2>
        <label htmlFor="describe">Describe your style in a sentence or two.</label>
      </h2>
      <p className="lede">
        Say it the way you would to a friend. This is the Stated You: it only knows what you write
        here.
      </p>
      <textarea
        id="describe"
        className="describe-box"
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={5}
        maxLength={1000}
        placeholder="Oversized earth tones, worn-in boots, nothing with a logo."
        autoFocus
      />
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      <div className="form-actions">
        <button className="btn" type="submit" disabled={busy || !text.trim()}>
          {busy ? 'Saving…' : 'Continue'}
        </button>
      </div>
    </form>
  )
}
