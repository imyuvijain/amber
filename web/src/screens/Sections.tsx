import { useState } from 'react'
import { ApiError, setFilters } from '../api'
import '../components/Form.css'
import './Sections.css'
import type { ScreenProps } from './props'

const GENDERS = ['Men', 'Women', 'Boys', 'Girls', 'Unisex']
const SECTIONS = ['Apparel', 'Footwear', 'Accessories']

function toggle(list: string[], value: string): string[] {
  return list.includes(value) ? list.filter((v) => v !== value) : [...list, value]
}

interface ChipsProps {
  legend: string
  options: string[]
  value: string[]
  onChange: (next: string[]) => void
}

function Chips({ legend, options, value, onChange }: ChipsProps) {
  return (
    <fieldset className="chips">
      <legend>{legend}</legend>
      <div className="chips-row">
        {options.map((o) => (
          <button
            key={o}
            type="button"
            className="chip"
            aria-pressed={value.includes(o)}
            onClick={() => onChange(toggle(value, o))}
          >
            {o}
          </button>
        ))}
      </div>
    </fieldset>
  )
}

export default function Sections({ userId, advance }: ScreenProps) {
  const [genders, setGenders] = useState<string[]>([])
  const [sections, setSections] = useState<string[]>(['Apparel', 'Footwear'])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function submit() {
    if (busy) return
    setBusy(true)
    setError(null)
    try {
      await setFilters(userId, genders, sections)
      advance()
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not save that. Try again.')
      setBusy(false)
    }
  }

  return (
    <div className="form-screen">
      <h2>What do you actually wear?</h2>
      <p className="lede">
        Every pair you see comes from what you tick here, so nothing arrives that you would never
        put on.
      </p>
      <Chips legend="Shop for" options={GENDERS} value={genders} onChange={setGenders} />
      <Chips legend="Including" options={SECTIONS} value={sections} onChange={setSections} />
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      <div className="form-actions">
        <button
          className="btn"
          type="button"
          onClick={submit}
          disabled={busy || genders.length === 0 || sections.length === 0}
        >
          {busy ? 'Saving…' : 'Start picking'}
        </button>
        {(genders.length === 0 || sections.length === 0) && (
          <span className="lede">Pick at least one of each.</span>
        )}
      </div>
    </div>
  )
}
