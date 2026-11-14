import { useCallback, useEffect, useState, type ComponentType } from 'react'
import Shell from './components/Shell'
import type { ScreenProps } from './screens/props'
import Quiz from './screens/Quiz'
import Results from './screens/Results'
import Sections from './screens/Sections'
import Showdown from './screens/Showdown'
import StyleWords from './screens/StyleWords'
import ThisOrThat from './screens/ThisOrThat'
import Topics from './screens/Topics'
import Welcome from './screens/Welcome'
import { clearSession, loadStep, loadUserId, saveStep, saveUserId } from './storage'
import { isStepId, stepAfter } from './steps'

// One screen per step id in steps.ts. A new step is one entry here and one in STEPS.
const SCREENS: Record<string, ComponentType<ScreenProps>> = {
  sections: Sections,
  style: StyleWords,
  quiz: Quiz,
  art: ThisOrThat,
  clothes: Showdown,
  results: Results,
}

function initial(): { userId: string | null; step: string } {
  const userId = loadUserId()
  const step = loadStep()
  if (userId && isStepId(step) && step !== 'welcome') return { userId, step }
  return { userId: null, step: 'welcome' }
}

// The veto screen (topics) is the front page; the closed v0 quiz flow lives at #v0.
function useHash(): string {
  const [hash, setHash] = useState(() => window.location.hash)
  useEffect(() => {
    const on = () => setHash(window.location.hash)
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  return hash
}

export default function App() {
  return useHash() === '#v0' ? <V0 /> : (
    <Shell step="welcome" signedIn={false} onReset={() => {}}>
      <Topics />
    </Shell>
  )
}

function V0() {
  const [{ userId, step }, setSession] = useState(initial)

  const go = useCallback((next: string) => {
    saveStep(next)
    setSession((s) => ({ ...s, step: next }))
  }, [])

  const reset = useCallback(() => {
    clearSession()
    setSession({ userId: null, step: 'welcome' })
  }, [])

  const created = (id: string) => {
    saveUserId(id)
    saveStep('sections')
    setSession({ userId: id, step: 'sections' })
  }

  const Screen = SCREENS[step]
  return (
    <Shell step={step} signedIn={userId !== null} onReset={reset}>
      {userId === null || !Screen ? (
        <Welcome onCreated={created} />
      ) : (
        <Screen userId={userId} advance={() => go(stepAfter(step))} reset={reset} />
      )}
    </Shell>
  )
}
