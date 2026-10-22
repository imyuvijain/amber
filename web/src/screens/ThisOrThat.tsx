import PairScreen from '../components/PairScreen'
import type { ScreenProps } from './props'

/** The art picks: this-or-that on artworks. They teach the Art You, nothing is scored yet. */
export default function ThisOrThat({ userId, advance, reset }: ScreenProps) {
  return (
    <PairScreen
      userId={userId}
      domain="art"
      phase="train"
      heading="Which would you rather hang on your wall?"
      hint="These are artworks. Go with your gut: pictures only, no titles or artists."
      progressLabel="Artworks picked"
      dwellMs={220}
      onDone={advance}
      onReset={reset}
    />
  )
}
