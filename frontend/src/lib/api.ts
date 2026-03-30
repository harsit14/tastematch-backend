import { useAuthStore } from './authStore'

const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

// Prevent multiple simultaneous refresh calls
let refreshPromise: Promise<string> | null = null

async function refreshAccessToken(): Promise<string> {
  if (refreshPromise) return refreshPromise

  refreshPromise = (async () => {
    const { refreshToken, updateTokens, clearAuth } = useAuthStore.getState()
    if (!refreshToken) {
      clearAuth()
      throw new ApiError(401, 'Session expired. Please sign in again.')
    }

    const res = await fetch(`${BASE_URL}/auth/refresh`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: refreshToken }),
    })

    if (!res.ok) {
      clearAuth()
      throw new ApiError(401, 'Session expired. Please sign in again.')
    }

    const data = await res.json()
    updateTokens(data.access_token, data.refresh_token, data.expires_in)
    return data.access_token as string
  })().finally(() => {
    refreshPromise = null
  })

  return refreshPromise
}

async function request<T>(
  path: string,
  options: RequestInit = {},
  token?: string
): Promise<T> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string>),
  }

  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }

  const res = await fetch(`${BASE_URL}${path}`, { ...options, headers })

  if (res.status === 401 && token) {
    // Token expired — try to refresh and retry once
    try {
      const newToken = await refreshAccessToken()
      const retryHeaders = { ...headers, Authorization: `Bearer ${newToken}` }
      const retry = await fetch(`${BASE_URL}${path}`, { ...options, headers: retryHeaders })

      if (!retry.ok) {
        const body = await retry.json().catch(() => ({}))
        throw new ApiError(retry.status, body.detail ?? 'Request failed')
      }
      return retry.json() as Promise<T>
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        // Refresh failed — clear auth so the guard redirects to login
        useAuthStore.getState().clearAuth()
      }
      throw err
    }
  }

  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new ApiError(res.status, body.detail ?? 'Request failed')
  }

  return res.json() as Promise<T>
}

export const api = {
  get: <T>(path: string, token?: string) =>
    request<T>(path, { method: 'GET' }, token),

  post: <T>(path: string, body: unknown, token?: string) =>
    request<T>(path, { method: 'POST', body: JSON.stringify(body) }, token),

  put: <T>(path: string, body: unknown, token?: string) =>
    request<T>(path, { method: 'PUT', body: JSON.stringify(body) }, token),

  delete: <T>(path: string, token?: string) =>
    request<T>(path, { method: 'DELETE' }, token),
}
