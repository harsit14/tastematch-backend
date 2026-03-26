-- Chat sessions group messages into conversations.
-- main.py creates sessions via POST /chat/session and queries them via GET /chat/sessions.
create table public.chat_sessions (
  id uuid default gen_random_uuid() primary key,
  user_id uuid references public.user_profiles(id) on delete cascade not null,
  title text not null default 'New conversation',
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create index chat_sessions_user_updated_at_idx
  on public.chat_sessions(user_id, updated_at desc);


-- Individual messages within a session.
create table public.chat_messages (
  id uuid default gen_random_uuid() primary key,
  session_id uuid references public.chat_sessions(id) on delete cascade not null,
  user_id uuid references public.user_profiles(id) on delete cascade not null,
  role text check (role in ('user', 'assistant')) not null,
  content text not null,
  created_at timestamptz default now()
);

-- main.py fetches history ordered by created_at for a given session
create index chat_messages_session_created_at_idx
  on public.chat_messages(session_id, created_at asc);