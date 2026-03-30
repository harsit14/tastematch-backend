import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { User } from '@/types'

interface AuthStore {
  user: User | null
  accessToken: string | null
  refreshToken: string | null
  tokenExpiry: number | null        // unix ms
  isAuthenticated: boolean
  setAuth: (user: User, accessToken: string, refreshToken: string, expiresIn: number) => void
  updateTokens: (accessToken: string, refreshToken: string, expiresIn: number) => void
  clearAuth: () => void
}

export const useAuthStore = create<AuthStore>()(
  persist(
    (set) => ({
      user: null,
      accessToken: null,
      refreshToken: null,
      tokenExpiry: null,
      isAuthenticated: false,

      setAuth: (user, accessToken, refreshToken, expiresIn) =>
        set({
          user,
          accessToken,
          refreshToken,
          // shave 60 s off so we refresh before the server rejects it
          tokenExpiry: Date.now() + (expiresIn - 60) * 1000,
          isAuthenticated: true,
        }),

      updateTokens: (accessToken, refreshToken, expiresIn) =>
        set({
          accessToken,
          refreshToken,
          tokenExpiry: Date.now() + (expiresIn - 60) * 1000,
        }),

      clearAuth: () =>
        set({ user: null, accessToken: null, refreshToken: null, tokenExpiry: null, isAuthenticated: false }),
    }),
    {
      name: 'tastematch-auth',
      partialize: (state) => ({
        user: state.user,
        accessToken: state.accessToken,
        refreshToken: state.refreshToken,
        tokenExpiry: state.tokenExpiry,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
)
