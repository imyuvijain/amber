import type { Domain } from '../api'
import PairScreen from '../components/PairScreen'
import type { ScreenProps } from './props'

const COPY: Record<Domain, { heading: string; hint: string; label: string }> = {
  clothes: {
    heading: 'Which would you rather wear?',
    hint: 'Go with your gut. Pictures only, no names or prices.',
    label: 'Clothes picked',
  },
  art: {
    heading: 'Which would you rather hang on your wall?',
    hint: 'These are artworks. Go with your gut: pictures only, no titles or artists.',
    label: 'Artworks picked',
  },
}

/** Training picks for one domain. Reused for art by passing `domain="art"`. */
export default function ThisOrThat({
  userId,
  advance,
  reset,
  domain = 'clothes',
}: ScreenProps & { domain?: Domain }) {
  const copy = COPY[domain]
  return (
    <PairScreen
      userId={userId}
      domain={domain}
      phase="train"
      heading={copy.heading}
      hint={copy.hint}
      progressLabel={copy.label}
      dwellMs={220}
      onDone={advance}
      onReset={reset}
    />
  )
}
