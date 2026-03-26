import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'
import { useToken } from './useToken'
import type { GlucoseReading } from '@/types'

/**
 * Fetches recent glucose readings.
 * @param limit  Max number of readings to fetch (default 20)
 */
export function useGlucose(limit = 20) {
  const token = useToken()

  return useQuery<GlucoseReading[]>({
    queryKey: ['glucose', limit],
    queryFn: () => api.get(`/glucose?limit=${limit}`, token),
    enabled: !!token,
  })
}
