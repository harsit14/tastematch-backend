create table public.fridge_items (
  id uuid primary key default gen_random_uuid(),

  -- References user_profiles, not auth.users directly, so cascade deletes
  -- chain correctly when a profile is removed.
  user_id uuid references public.user_profiles(id) on delete cascade not null,

  -- food_id is optional — user can add a freeform item without a foods table match.
  food_id uuid references public.foods(id) on delete set null,

  -- Human-readable name always stored so the fridge is usable even when
  -- food_id is null or the foods row is later deleted.
  food_name text not null,

  quantity numeric,          -- e.g. 200
  unit text,                 -- 'g', 'oz', 'cups', 'pieces'
  expiry_date date,
  added_at timestamptz default now()
);

create index fridge_items_user_id_idx on public.fridge_items(user_id);