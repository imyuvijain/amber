// The five Yous. Defined once: id (as the API names it), display name, color, one-line pitch.
// Colors are CSS custom properties (see index.css) so they follow light/dark; later screens
// (the map) read the same `var(--you-*)`.

export interface You {
  id: string
  name: string
  color: string
  pitch: string
}

export const YOUS: readonly You[] = [
  { id: 'stated', name: 'Stated you', color: 'var(--you-stated)', pitch: 'Who you say you are.' },
  { id: 'bigfive', name: 'Big Five you', color: 'var(--you-bigfive)', pitch: 'Who your personality profile says you are.' },
  { id: 'art', name: 'Art you', color: 'var(--you-art)', pitch: 'Who your taste in paintings says you are.' },
  { id: 'values', name: 'Values you', color: 'var(--you-values)', pitch: 'Who your principles say you are.' },
  { id: 'revealed', name: 'Revealed you', color: 'var(--you-revealed)', pitch: 'Who your clicks say you are.' },
]

const FALLBACK: Omit<You, 'id'> = { name: 'Unknown you', color: 'var(--muted)', pitch: '' }

export const youById = (id: string): You => YOUS.find((y) => y.id === id) ?? { id, ...FALLBACK }
