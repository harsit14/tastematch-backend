import { Navigate, Outlet } from 'react-router-dom'
import { useAuthStore } from '@/lib/authStore'

export default function AuthGuard() {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated)
  return isAuthenticated ? <Outlet /> : <Navigate to="/login" replace />
}
