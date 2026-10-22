import PairScreen from '../components/PairScreen'
import type { ScreenProps } from './props'

/**
 * The only clothes screen: every pick is scored. Your Yous guess each pair before you
 * answer, and the tally below keeps count after every pick.
 */
export default function Showdown({ userId, advance, reset }: ScreenProps) {
  return (
    <PairScreen
      userId={userId}
      domain="clothes"
      phase="showdown"
      heading="Which would you rather wear?"
      hint="Go with your gut. Every pick counts: your Yous guessed it before you answered."
      progressLabel="Clothes picked"
      showCalls
      dwellMs={1500}
      onDone={advance}
      onReset={reset}
    />
  )
}
