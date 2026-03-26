# TasteMatch Frontend

React/TypeScript frontend for the TasteMatch nutrition intelligence platform.

## Tech stack

- React 18 + TypeScript
- Vite
- React Router v6
- TanStack Query v5
- Zustand (auth state)
- Framer Motion (animations)
- Recharts (data visualisation)
- CSS Modules (component scoped styles)

## Getting started

```bash
cp .env.example .env
# Set VITE_API_URL to your backend URL

npm install
npm run dev
```

## Project structure

```
src/
  components/
    layout/         # AppLayout, Sidebar, PageHeader, AuthGuard
    ui/             # Shared primitives: Button, Input, Card, Badge, Modal, StatCard, EmptyState
    features/
      auth/         # LoginPage, SignupPage
      dashboard/    # DashboardPage
      glucose/      # GlucosePage
      meals/        # MealsPage
      fridge/       # FridgePage
      recipes/      # RecipesPage
      chat/         # ChatPage
      metrics/      # MetricsPage
      profile/      # ProfilePage
  hooks/            # Custom hooks (reserved)
  lib/
    api.ts          # Typed fetch client
    authStore.ts    # Zustand auth store with persistence
    utils.ts        # Date, glucose, BMI helpers
  types/
    index.ts        # All TypeScript interfaces
  styles/
    globals.css     # Design tokens + reset
```

## Design system

All colours, spacing and typography use CSS custom properties defined in `globals.css`.
Never hardcode colour values in component stylesheets — reference variables directly.

| Token | Value |
|-------|-------|
| `--jungle-teal` | `#6b9080` — primary action colour |
| `--muted-teal` | `#a4c3b2` — secondary / accents |
| `--frozen-water` | `#cce3de` — borders, backgrounds |
| `--azure-mist` | `#eaf4f4` — sunken surfaces |
| `--mint-cream` | `#f6fff8` — page background |

## Authentication

JWT access token is stored in `localStorage` via Zustand persist middleware.
All authenticated API calls pass the token as `Authorization: Bearer <token>`.
