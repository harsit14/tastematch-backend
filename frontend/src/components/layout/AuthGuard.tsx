import { useEffect, useState } from 'react'
import { Navigate, Outlet } from 'react-router-dom'
import { useAuthStore } from '@/lib/authStore'

export default function AuthGuard() {
  const { isAuthenticated, tokenExpiry, refreshToken, updateTokens, clearAuth } = useAuthStore()
  const [checking, setChecking] = useState(true)

  useEffect(() => {
    if (!isAuthenticated) {
      setChecking(false)
      return
    }

    // If the token is already expired (or will expire in the next 60 s), refresh now
    if (tokenExpiry && Date.now() >= tokenExpiry) {
      if (!refreshToken) {
        clearAuth()
        setChecking(false)
        return
      }

      fetch(`${import.meta.env.VITE_API_URL ?? 'http://localhost:8000'}/auth/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: refreshToken }),
      })
        .then((res) => (res.ok ? res.json() : Promise.reject(res.status)))
        .then((data) => updateTokens(data.access_token, data.refresh_token, data.expires_in))
        .catch(() => clearAuth())
        .finally(() => setChecking(false))
    } else {
      setChecking(false)
    }
  }, [])  // eslint-disable-line react-hooks/exhaustive-deps

  if (checking) return null   // brief flicker-free pause while refreshing

  return isAuthenticated ? <Outlet /> : <Navigate to="/login" replace state={{ message: 'Your session expired. Please sign in again.' }} />
}
