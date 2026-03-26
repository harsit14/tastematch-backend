create table public.user_profiles (
  id uuid references auth.users(id) on delete cascade primary key,
  first_name text not null,
  last_name text not null,
  age int not null,
  diabetes_type text check (
    diabetes_type in ('type1', 'type2', 'prediabetes', 'gestational', 'other', 'none')
  ) not null,
  date_of_birth date not null,
  dietary_preferences text[],  -- e.g. ['vegetarian', 'gluten-free']
  allergies text[],
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);