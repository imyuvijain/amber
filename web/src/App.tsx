import { useCallback, useState, type ComponentType } from 'react'
import Shell from './components/Shell'
import type { ScreenProps } from './screens/props'
import Describe from './screens/Describe'
import Scoreboard from './screens/Scoreboard'
import Sections from './screens/Sections'
import Showdown from './screens/Showdown'
import ThisOrThat from './screens/ThisOrThat'
import Welcome from './screens/Welcome'
import { clearSession, loadStep, loadUserId, saveStep, saveUserId } from './storage'
import { isStepId, stepAfter } from './steps'

// One screen per step id in steps.ts. A new step is one entry here and one in STEPS.
const SCREENS: Record<string, ComponentType<ScreenProps>> = {
  describe: Describe,
  sections: Sections,
  train: ThisOrThat,
  showdown: Showdown,
  scoreboard: Scoreboard,
}

function initial(): { userId: string | null; step: string } {
  const userId = loadUserId()
  const step = loadStep()
  if (userId && isStepId(step) && step !== 'welcome') return { userId, step }
  return { userId: null, step: 'welcome' }
}

export default function App() {
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
    saveStep('describe')
    setSession({ userId: id, step: 'describe' })
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
