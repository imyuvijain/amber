import { useCallback, useEffect, useRef, useState } from 'react'
import {
  ApiError,
  choose as chooseApi,
  nextPair,
  type Domain,
  type Item,
  type Pair,
  type Phase,
  type Progress,
  type Reveal,
} from '../api'

export type Status = 'loading' | 'ready' | 'sending' | 'showing'
export type Tally = Record<string, { hit: number; n: number }>

interface Options {
  userId: string
  domain: Domain
  phase: Phase
  /** How long the pick (and, in a showdown, the reveal) stays on screen before the next pair. */
  dwellMs: number
  onDone: () => void
}

interface Failure {
  message: string
  /** 'load' failures offer a retry; 'choose' failures just re-enable the buttons. */
  kind: 'load' | 'choose'
  status: number
}

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms))

/** Resolves once both images are in the cache (or failed, or `capMs` passed). */
function preload(items: Item[], capMs = 2500): Promise<void> {
  const loads = items.map(
    (item) =>
      new Promise<void>((resolve) => {
        const img = new Image()
        img.onload = img.onerror = () => resolve()
        img.src = item.thumb_url
      }),
  )
  return Promise.race([Promise.all(loads).then(() => undefined), sleep(capMs)])
}

function describe(err: unknown, fallback: string): { message: string; status: number } {
  return err instanceof ApiError
    ? { message: err.message, status: err.status }
    : { message: fallback, status: 0 }
}

/**
 * Drives one run of pairs: load the open pair, send a pick, fetch and preload the next pair
 * while the pick/reveal is on screen. `busy` is a ref so a double-click in the same frame is
 * ignored before React re-renders.
 */
export function usePairFlow({ userId, domain, phase, dwellMs, onDone }: Options) {
  const [pair, setPair] = useState<Pair | null>(null)
  const [progress, setProgress] = useState<Progress | null>(null)
  const [status, setStatus] = useState<Status>('loading')
  const [picked, setPicked] = useState<string | null>(null)
  const [reveal, setReveal] = useState<Reveal | null>(null)
  const [tally, setTally] = useState<Tally>({})
  const [failure, setFailure] = useState<Failure | null>(null)

  const busy = useRef(true)
  // Bumped on unmount (and on StrictMode's remount) so stale async work never touches state.
  const epoch = useRef(0)
  const onDoneRef = useRef(onDone)
  useEffect(() => {
    onDoneRef.current = onDone
  })

  /** Fetch the open pair (also used to retry). Returns false when the run is finished. */
  const load = useCallback(async () => {
    const e = epoch.current
    busy.current = true
    setFailure(null)
    setStatus('loading')
    try {
      const next = await nextPair(userId, domain, phase)
      if (e !== epoch.current) return
      if (next.done) {
        onDoneRef.current()
        return
      }
      await preload([next.a, next.b])
      if (e !== epoch.current) return
      setPair(next)
      setProgress(next.progress)
      setPicked(null)
      setReveal(null)
      setStatus('ready')
      busy.current = false
    } catch (err) {
      if (e !== epoch.current) return
      setFailure({ ...describe(err, 'Could not load the next pair.'), kind: 'load' })
    }
  }, [userId, domain, phase])

  useEffect(() => {
    void load()
    return () => {
      epoch.current++
    }
  }, [load])

  const pick = useCallback(
    async (item: Item) => {
      if (busy.current || !pair) return
      const e = epoch.current
      busy.current = true
      setPicked(item.id)
      setStatus('sending')
      setFailure(null)
      let result
      try {
        result = await chooseApi(pair.pair_id, item.id)
      } catch (err) {
        if (e !== epoch.current) return
        setPicked(null)
        setStatus('ready')
        setFailure({ ...describe(err, 'Could not save that pick. Try again.'), kind: 'choose' })
        busy.current = false
        return
      }
      if (e !== epoch.current) return
      setProgress(result.progress)
      setReveal(result.reveal)
      if (result.reveal) {
        const calls = result.reveal
        setTally((t) => {
          const next = { ...t }
          for (const [you, hit] of Object.entries(calls)) {
            const row = next[you] ?? { hit: 0, n: 0 }
            next[you] = { hit: row.hit + (hit ? 1 : 0), n: row.n + 1 }
          }
          return next
        })
      }
      setStatus('showing')
      // The pick is saved; from here a failure is about loading, not choosing.
      busy.current = true
      try {
        const [next] = await Promise.all([nextPair(userId, domain, phase), sleep(dwellMs)])
        if (e !== epoch.current) return
        if (next.done) {
          onDoneRef.current()
          return
        }
        await preload([next.a, next.b])
        if (e !== epoch.current) return
        setPair(next)
        setProgress(next.progress)
        setPicked(null)
        setReveal(null)
        setStatus('ready')
        busy.current = false
      } catch (err) {
        if (e !== epoch.current) return
        setFailure({ ...describe(err, 'Could not load the next pair.'), kind: 'load' })
      }
    },
    [pair, userId, domain, phase, dwellMs],
  )

  return { pair, progress, status, picked, reveal, tally, failure, pick, retry: load }
}
