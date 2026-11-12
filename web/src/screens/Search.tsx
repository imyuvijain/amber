import { useEffect, useRef, useState, type CSSProperties, type FormEvent } from 'react'
import { ApiError, search, type Item, type SearchResults } from '../api'
import '../components/Form.css'
import './Sections.css' // the .chips / .chip look
import { YOUS, youById } from '../yous'
import './Search.css'

// Tap-only by default: these cover the search; the box below is the one place typing is allowed.
const SUGGESTIONS = ['black jacket', 'summer dress', 'sneakers', 'white shirt', 'party wear', 'running shoes']

/** "Art you" -> "art", "Big Five you" -> "big five": the You's name as it reads in "your ___ taste". */
const tasteOf = (id: string) => youById(id).name.replace(/ you$/, '').toLowerCase()

function Grid({ items }: { items: Item[] }) {
  return (
    <ul className="hits">
      {items.map((item) => (
        <li key={item.id}>
          <img src={item.thumb_url} alt={item.title} title={item.title} loading="lazy" />
        </li>
      ))}
    </ul>
  )
}

interface ColumnProps {
  heading: string
  color?: string
  items: Item[]
  empty: string
}

function Column({ heading, color, items, empty }: ColumnProps) {
  return (
    <section className="hits-col" style={color ? ({ '--c': color } as CSSProperties) : undefined}>
      <h3>
        {color && <span className="bead" />}
        {heading}
      </h3>
      {items.length ? <Grid items={items} /> : <p className="note">{empty}</p>}
    </section>
  )
}

interface Props {
  userId: string
  onBack: () => void
}

/** Search the clothes you may see. Everyone's ranking sits next to the same results ranked by
 *  one of your Yous (picked with the chips; Art you to start with). */
export default function Search({ userId, onBack }: Props) {
  const [query, setQuery] = useState<string | null>(null)
  const [typed, setTyped] = useState('')
  const [you, setYou] = useState('art')
  const [found, setFound] = useState<SearchResults | null>(null)
  const [error, setError] = useState<string | null>(null)
  const latest = useRef(0) // a slower, older answer never overwrites a newer one

  useEffect(() => {
    if (query === null) return
    const mine = ++latest.current
    search(userId, query, you)
      .then((r) => {
        if (mine !== latest.current) return
        setError(null)
        setFound(r)
      })
      .catch((err) => {
        if (mine !== latest.current) return
        setError(err instanceof ApiError ? err.message : 'Could not search. Try again.')
      })
  }, [userId, query, you])

  const run = (q: string) => {
    setFound(null)
    setQuery(q)
  }
  const submit = (e: FormEvent) => {
    e.preventDefault()
    const q = typed.trim()
    if (q) run(q)
  }

  const picked = youById(you)
  const loading = query !== null && found === null && error === null
  return (
    <div className="search">
      <header className="search-head">
        <h2>Search the clothes</h2>
        <p className="lede">
          The same search twice: once as everyone would see it, once reordered by how one of your
          Yous would rank what it finds. Only sections you ticked show up.
        </p>
      </header>

      <fieldset className="chips">
        <legend>Look for</legend>
        <div className="chips-row">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              type="button"
              className="chip"
              aria-pressed={query === s}
              onClick={() => run(s)}
            >
              {s}
            </button>
          ))}
        </div>
        <form className="box" onSubmit={submit}>
          <label className="visually-hidden" htmlFor="search-box">
            Or search for something else
          </label>
          <input
            id="search-box"
            type="search"
            placeholder="Or type your own"
            maxLength={100}
            value={typed}
            onChange={(e) => setTyped(e.target.value)}
          />
          <button className="btn btn-quiet" type="submit" disabled={!typed.trim()}>
            Search
          </button>
        </form>
      </fieldset>

      <fieldset className="chips">
        <legend>Rank by</legend>
        <div className="chips-row">
          {YOUS.map((y) => (
            <button
              key={y.id}
              type="button"
              className="chip you-chip"
              style={{ '--c': y.color } as CSSProperties}
              aria-pressed={you === y.id}
              onClick={() => setYou(y.id)}
            >
              <span className="bead" />
              {y.name}
            </button>
          ))}
        </div>
      </fieldset>

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      {query === null && <p className="lede">Pick something to look for.</p>}
      {loading && <p className="lede">Searching…</p>}
      {found && (
        <div className="result-cols" aria-live="polite">
          <Column heading="Everyone" items={found.generic} empty="Nothing matches in your sections." />
          <Column
            heading={`Ranked by your ${tasteOf(found.you)} taste`}
            color={youById(found.you).color}
            items={found.personal}
            empty={
              found.generic.length
                ? `${picked.name} has nothing to go on yet, so it cannot rank these.`
                : 'Nothing matches in your sections.'
            }
          />
        </div>
      )}

      <div className="form-actions">
        <button className="btn btn-quiet" type="button" onClick={onBack}>
          Back to results
        </button>
      </div>
    </div>
  )
}
