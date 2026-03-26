import { createBrowserRouter, RouterProvider, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ToastProvider } from '@/components/ui/ToastProvider'
import AppLayout from '@/components/layout/AppLayout'
import AuthGuard from '@/components/layout/AuthGuard'
import LoginPage from '@/components/features/auth/LoginPage'
import SignupPage from '@/components/features/auth/SignupPage'
import DashboardPage from '@/components/features/dashboard/DashboardPage'
import GlucosePage from '@/components/features/glucose/GlucosePage'
import MealsPage from '@/components/features/meals/MealsPage'
import FridgePage from '@/components/features/fridge/FridgePage'
import RecipesPage from '@/components/features/recipes/RecipesPage'
import ChatPage from '@/components/features/chat/ChatPage'
import MetricsPage from '@/components/features/metrics/MetricsPage'
import ProfilePage from '@/components/features/profile/ProfilePage'
import FoodsPage from '@/components/features/foods/FoodsPage'
import NotFoundPage from '@/components/features/NotFoundPage'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 2,
      retry: 1,
    },
  },
})

const router = createBrowserRouter([
  {
    path: '/login',
    element: <LoginPage />,
  },
  {
    path: '/signup',
    element: <SignupPage />,
  },
  {
    element: <AuthGuard />,
    children: [
      {
        element: <AppLayout />,
        children: [
          { path: '/dashboard', element: <DashboardPage /> },
          { path: '/glucose', element: <GlucosePage /> },
          { path: '/meals', element: <MealsPage /> },
          { path: '/fridge', element: <FridgePage /> },
          { path: '/recipes', element: <RecipesPage /> },
          { path: '/chat', element: <ChatPage /> },
          { path: '/metrics', element: <MetricsPage /> },
          { path: '/profile', element: <ProfilePage /> },
          { path: '/foods', element: <FoodsPage /> },
        ],
      },
    ],
  },
  {
    path: '/',
    element: <Navigate to="/dashboard" replace />,
  },
  {
    path: '*',
    element: <NotFoundPage />,
  },
])

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ToastProvider>
        <RouterProvider router={router} />
      </ToastProvider>
    </QueryClientProvider>
  )
}
