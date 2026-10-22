export interface ScreenProps {
  userId: string
  /** Move to the next step in STEPS. */
  advance: () => void
  /** Forget this user and return to the welcome screen. */
  reset: () => void
}
