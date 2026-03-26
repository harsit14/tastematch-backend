create table public.foods (
  id uuid primary key default gen_random_uuid(),
  external_id text,
  source text,               -- 'usda', 'openfoodfacts', 'manual', etc.
  name text not null,
  brand text,
  calories_per_100g numeric,
  protein_per_100g numeric,
  carbs_per_100g numeric,
  fat_per_100g numeric,
  fiber_per_100g numeric,
  serving_size_g numeric,

  -- GI is lab-measured. USDA and Open Food Facts do not expose it.
  -- Will be null for auto-imported foods. Never default to 0.
  glycemic_index numeric check (glycemic_index >= 0 and glycemic_index <= 100),

  -- GL per standard serving = (glycemic_index * carbs_per_serving_g) / 100.
  -- Pre-computed convenience column. Low <10, Medium 10-20, High >20.
  glycemic_load_per_serving numeric check (glycemic_load_per_serving >= 0),

  -- Derived from glycemic_index for fast filtering without in-query math.
  -- Low: GI <= 55  |  Medium: 56-69  |  High: >= 70
  gi_category text check (gi_category in ('low', 'medium', 'high')),

  created_at timestamptz default now()
);

-- Prevents duplicate imports from the same external source
create unique index foods_external_id_source_idx
  on public.foods(external_id, source)
  where external_id is not null;

-- Full-text search index used by /foods/search
create index foods_name_fts_idx
  on public.foods using gin(to_tsvector('english', name));