import { useState, useEffect } from 'react'

/**
 * Delays updating a value until the user stops typing.
 * Used in the food search input to avoid firing a request on every keystroke.
 */
export function useDebounce<T>(value: T, delayMs = 400): T {
  const [debounced, setDebounced] = useState<T>(value)

  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delayMs)
    return () => clearTimeout(timer)
  }, [value, delayMs])

  return debounced
}
