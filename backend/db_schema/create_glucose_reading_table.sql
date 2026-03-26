create table public.glucose_readings (
  id uuid default gen_random_uuid() primary key,
  user_id uuid references public.user_profiles(id) on delete cascade not null,
  reading_mmol numeric not null,  -- stored in mmol/L, convert in app if needed
  reading_context text check (
    reading_context in ('fasting', 'before_meal', 'after_meal', 'bedtime', 'other')
  ),
  notes text,
  recorded_at timestamptz not null,
  created_at timestamptz default now()
);

create index glucose_readings_user_recorded_at_idx
  on public.glucose_readings(user_id, recorded_at desc);