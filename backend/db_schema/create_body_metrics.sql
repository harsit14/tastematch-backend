create table public.body_metrics (
  id uuid primary key default gen_random_uuid(),

  -- Consistent with all other tables: reference user_profiles, not auth.users.
  -- This ensures cascade deletes chain correctly and RLS policies are uniform.
  user_id uuid references public.user_profiles(id) on delete cascade not null,

  weight_kg numeric not null,
  height_cm numeric not null,

  -- Computed and stored by the database — no risk of app-layer rounding drift.
  bmi numeric generated always as (
    round(weight_kg / ((height_cm / 100.0) * (height_cm / 100.0)), 1)
  ) stored,

  recorded_at timestamptz default now()
);

create index body_metrics_user_recorded_at_idx
  on public.body_metrics(user_id, recorded_at desc);