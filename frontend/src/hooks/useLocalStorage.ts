import { useState, useCallback } from 'react'

/**
 * useState but backed by localStorage.
 * Useful for persisting non-auth UI preferences (e.g. sidebar state, filter choices).
 */
export function useLocalStorage<T>(key: string, initialValue: T) {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key)
      return item ? (JSON.parse(item) as T) : initialValue
    } catch {
      return initialValue
    }
  })

  const setValue = useCallback(
    (value: T | ((prev: T) => T)) => {
      try {
        const next = value instanceof Function ? value(storedValue) : value
        setStoredValue(next)
        window.localStorage.setItem(key, JSON.stringify(next))
      } catch {
        // Silently ignore write failures (private mode, storage full)
      }
    },
    [key, storedValue]
  )

  return [storedValue, setValue] as const
}
