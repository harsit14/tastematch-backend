create table public.saved_recipes (
  id uuid default gen_random_uuid() primary key,
  user_id uuid references public.user_profiles(id) on delete cascade not null,
  title text not null,
  ingredients jsonb not null,     -- array of {name, quantity, unit}
  instructions text not null,
  carbs_per_serving numeric,
  calories_per_serving numeric,
  servings integer,
  tags text[],                    -- e.g. ['low-carb', 'quick', 'low-gi']
  chroma_doc_id text,             -- reference back to Chroma vector if embedded
  created_at timestamptz default now()
);

create index saved_recipes_user_id_idx
  on public.saved_recipes(user_id);