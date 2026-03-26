create table public.meal_logs (
  id uuid default gen_random_uuid() primary key,
  user_id uuid references public.user_profiles(id) on delete cascade not null,
  meal_name text,
  carbs_grams numeric not null,
  calories numeric,
  meal_type text check (meal_type in ('breakfast', 'lunch', 'dinner', 'snack')),

  -- Estimated GI for the meal as a whole (0-100).
  -- Mixed meals have a lower effective GI than individual ingredients in isolation.
  -- NULL means unknown — do not default to 0.
  estimated_gi numeric check (estimated_gi >= 0 and estimated_gi <= 100),

  -- GL = (estimated_gi * carbs_grams) / 100.
  -- Computed and stored at insert time. Low <10, Medium 10-20, High >20.
  -- Must be recomputed if carbs_grams or estimated_gi are ever updated.
  glycemic_load numeric check (glycemic_load >= 0),

  notes text,
  eaten_at timestamptz not null,
  created_at timestamptz default now()
);

-- Most common query pattern: a user's meals sorted by time
create index meal_logs_user_eaten_at_idx
  on public.meal_logs(user_id, eaten_at desc);