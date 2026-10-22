import PairScreen from '../components/PairScreen'
import type { ScreenProps } from './props'

/** The test. Same layout as training, but every You has already guessed. Nothing here trains. */
export default function Showdown({ userId, advance, reset }: ScreenProps) {
  return (
    <PairScreen
      userId={userId}
      domain="clothes"
      phase="showdown"
      heading="Showdown: pick again."
      hint="Your Yous guessed these before you answered. Nothing you pick here teaches them."
      progressLabel="Showdown pairs picked"
      showCalls
      dwellMs={1500}
      onDone={advance}
      onReset={reset}
    />
  )
}
