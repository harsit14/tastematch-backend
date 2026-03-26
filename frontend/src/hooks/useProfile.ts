import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'
import { useToken } from './useToken'
import type { UserProfile } from '@/types'

/**
 * Fetches the current user's profile. Shared across Dashboard, Chat context builder, etc.
 * Uses staleTime from the QueryClient default (2 minutes).
 */
export function useProfile() {
  const token = useToken()

  return useQuery<UserProfile>({
    queryKey: ['profile'],
    queryFn: () => api.get('/profile', token),
    enabled: !!token,
  })
}
