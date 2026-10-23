// Typed client for the Amber API (src/amber/api/app.py). One function per route.

export type Domain = 'clothes' | 'art'
// Art uses 'train'; clothes use 'showdown' (the one scored phase).
export type Phase = 'train' | 'showdown' | 'retest'

export interface Item {
  id: string
  domain: Domain
  title: string
  thumb_url: string
  labels: Record<string, string>
}

export interface Progress {
  done: number
  total: number
}

export interface Pair {
  done?: false
  pair_id: string
  a: Item
  b: Item
  progress: Progress
}

export interface Finished {
  done: true
  progress: Progress
}

/** For clothes pairs: whether each You called it. Abstaining Yous are absent. */
export type Reveal = Record<string, boolean>

export interface Chosen {
  progress: Progress
  reveal: Reveal | null
}

export interface Score {
  you: string
  correct: number
  n: number
  accuracy: number
  ci_low: number
  ci_high: number
}

export interface Results {
  /** Every clothes pick. */
  scores_all: Score[]
  /** The second half of the picks (21-40 of 40). */
  scores_late: Score[]
  /** Rolling accuracy per pick, by You. */
  curves: Record<string, number[]>
  n_clothes: number
}

/** An HTTP error carrying the server's `detail` and status. */
export class ApiError extends Error {
  readonly status: number
  constructor(status: number, detail: string) {
    super(detail)
    this.name = 'ApiError'
    this.status = status
  }
}

function detailOf(body: unknown, fallback: string): string {
  const detail = (body as { detail?: unknown } | null)?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail.map((d) => (d as { msg?: string }).msg ?? '').filter(Boolean).join('; ') || fallback
  }
  return fallback
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  let res: Response
  try {
    res = await fetch(path, {
      method,
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    })
  } catch {
    throw new ApiError(0, 'Cannot reach the Amber server. Is `uv run amber serve` running?')
  }
  const payload: unknown = await res.json().catch(() => null)
  if (!res.ok) throw new ApiError(res.status, detailOf(payload, `${res.status} ${res.statusText}`))
  return payload as T
}

export const createUser = (name: string) =>
  request<{ user_id: string }>('POST', '/api/users', { name })

export const setFilters = (uid: string, genders: string[], sections: string[]) =>
  request<{ allowed: number }>('PUT', `/api/users/${uid}/filters`, { genders, sections })

export const getStyleWords = () => request<{ rows: string[][] }>('GET', '/api/style-words')

export const putStyle = (uid: string, words: string[]) =>
  request<Record<string, never>>('PUT', `/api/users/${uid}/style`, { words })

export const nextPair = (uid: string, domain: Domain, phase: Phase) =>
  request<Pair | Finished>('POST', `/api/users/${uid}/next`, { domain, phase })

export const choose = (pairId: string, chosenId: string) =>
  request<Chosen>('POST', `/api/pairs/${pairId}/choose`, { chosen_id: chosenId })

export const getResults = (uid: string) => request<Results>('GET', `/api/users/${uid}/results`)

/** Warm the browser cache so the next pair appears instantly. */
export function prefetchImages(...items: Item[]): void {
  for (const item of items) new Image().src = item.thumb_url
}
