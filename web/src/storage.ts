// localStorage can throw (private windows, blocked site data), so every access is guarded.

const USER_KEY = 'amber.user_id'
const STEP_KEY = 'amber.step'

function read(key: string): string | null {
  try {
    return localStorage.getItem(key)
  } catch {
    return null
  }
}

function write(key: string, value: string | null): void {
  try {
    if (value === null) localStorage.removeItem(key)
    else localStorage.setItem(key, value)
  } catch {
    // Not persisting just means a reload starts over.
  }
}

export const loadUserId = () => read(USER_KEY)
export const saveUserId = (id: string) => write(USER_KEY, id)
export const loadStep = () => read(STEP_KEY)
export const saveStep = (step: string) => write(STEP_KEY, step)

export function clearSession(): void {
  write(USER_KEY, null)
  write(STEP_KEY, null)
}
