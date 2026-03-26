import { useAuthStore } from '@/lib/authStore'

/**
 * Returns the access token string or undefined.
 * Prefer this over calling useAuthStore everywhere — keeps token access consistent.
 */
export function useToken(): string | undefined {
  return useAuthStore((s) => s.accessToken) ?? undefined
}
