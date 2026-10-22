// The flow, in order. Later tasks insert their steps (Big Five, values, art) before 'train'
// by adding an entry here and a screen in App.tsx's SCREENS map.

export interface Step {
  id: string
  label: string
}

export const STEPS: readonly Step[] = [
  { id: 'welcome', label: 'Start' },
  { id: 'describe', label: 'Describe' },
  { id: 'sections', label: 'Sections' },
  { id: 'art', label: 'Art' },
  { id: 'train', label: 'Pick' },
  { id: 'showdown', label: 'Showdown' },
  { id: 'scoreboard', label: 'Results' },
]

export const isStepId = (s: string | null): s is string => STEPS.some((x) => x.id === s)

/** The step after `id`, or `id` itself at the end. */
export function stepAfter(id: string): string {
  const i = STEPS.findIndex((s) => s.id === id)
  return STEPS[Math.min(i + 1, STEPS.length - 1)].id
}
