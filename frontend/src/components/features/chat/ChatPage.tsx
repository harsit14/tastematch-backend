import { useState, useRef, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import { useAuthStore } from '@/lib/authStore'
import { api } from '@/lib/api'
import { formatTime } from '@/lib/utils'
import type { ChatSession, ChatMessage } from '@/types'
import Button from '@/components/ui/Button'
import styles from './ChatPage.module.css'

export default function ChatPage() {
  const token = useAuthStore((s) => s.accessToken)
  const qc = useQueryClient()
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const { data: sessions } = useQuery<ChatSession[]>({
    queryKey: ['chat-sessions'],
    queryFn: () => api.get('/chat/sessions', token ?? undefined),
    enabled: !!token,
  })

  const { data: sessionMessages } = useQuery<ChatMessage[]>({
    queryKey: ['chat-messages', activeSessionId],
    queryFn: () => api.get(`/chat/sessions/${activeSessionId}/messages`, token ?? undefined),
    enabled: !!token && !!activeSessionId,
    staleTime: 0,
  })

  useEffect(() => {
    if (sessionMessages) {
      setMessages(sessionMessages)
    }
  }, [sessionMessages])

  const createSessionMutation = useMutation({
    mutationFn: () => api.post<{ session_id: string; title: string }>('/chat/session', { title: 'New conversation' }, token ?? undefined),
    onSuccess: (data) => {
      qc.invalidateQueries({ queryKey: ['chat-sessions'] })
      setActiveSessionId(data.session_id)
      setMessages([])
    },
  })

  const sendMutation = useMutation({
    mutationFn: (body: object) => api.post<{ response: string }>('/chat/message', body, token ?? undefined),
    onSuccess: (data) => {
      const assistantMsg: ChatMessage = {
        id: crypto.randomUUID(),
        session_id: activeSessionId!,
        role: 'assistant',
        content: data.response,
        created_at: new Date().toISOString(),
      }
      setMessages((prev) => [...prev, assistantMsg])
      setIsTyping(false)
    },
    onError: () => setIsTyping(false),
  })

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isTyping])

  const handleSend = async () => {
    if (!input.trim() || !activeSessionId) return

    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      session_id: activeSessionId,
      role: 'user',
      content: input.trim(),
      created_at: new Date().toISOString(),
    }

    setMessages((prev) => [...prev, userMsg])
    const messageText = input.trim()
    setInput('')
    setIsTyping(true)

    sendMutation.mutate({ session_id: activeSessionId, message: messageText })
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleNewChat = () => {
    createSessionMutation.mutate()
  }

  const handleSelectSession = (sessionId: string) => {
    setActiveSessionId(sessionId)
    setMessages([])
  }

  return (
    <div className={styles.container}>
      <aside className={styles.sessionsSidebar}>
        <div className={styles.sessionsHeader}>
          <h2 className={styles.sessionsTitle}>Conversations</h2>
          <Button size="sm" onClick={handleNewChat} loading={createSessionMutation.isPending}>
            New chat
          </Button>
        </div>

        {sessions && sessions.length > 0 ? (
          <ul className={styles.sessionList}>
            {sessions.map((session) => (
              <li key={session.id}>
                <button
                  className={`${styles.sessionItem} ${activeSessionId === session.id ? styles.sessionItemActive : ''}`}
                  onClick={() => handleSelectSession(session.id)}
                >
                  <span className={styles.sessionTitle}>{session.title}</span>
                  <span className={styles.sessionDate}>
                    {new Date(session.updated_at).toLocaleDateString('en-GB', { day: 'numeric', month: 'short' })}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <p className={styles.noSessions}>No conversations yet. Start a new chat.</p>
        )}
      </aside>

      <div className={styles.chatArea}>
        {!activeSessionId ? (
          <div className={styles.welcomeState}>
            <div className={styles.welcomeIcon}>
              <LeafIcon />
            </div>
            <h2 className={styles.welcomeTitle}>Ask TasteMatch</h2>
            <p className={styles.welcomeDesc}>
              Your intelligent nutrition assistant understands your health profile, glucose patterns,
              and what is in your fridge. Ask anything about meals, recipes, or managing your blood sugar.
            </p>
            <div className={styles.welcomeSuggestions}>
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  className={styles.suggestionChip}
                  onClick={async () => {
                    const session = await createSessionMutation.mutateAsync()
                    const userMsg: ChatMessage = {
                      id: crypto.randomUUID(),
                      session_id: session.session_id,
                      role: 'user',
                      content: s,
                      created_at: new Date().toISOString(),
                    }
                    setMessages([userMsg])
                    setIsTyping(true)
                    sendMutation.mutate({ session_id: session.session_id, message: s })
                  }}
                >
                  {s}
                </button>
              ))}
            </div>
            <Button size="lg" onClick={handleNewChat} loading={createSessionMutation.isPending}>
              Start a conversation
            </Button>
          </div>
        ) : (
          <>
            <div className={styles.messages}>
              {messages.length === 0 && (
                <div className={styles.emptyMessages}>
                  <p>Send a message to get started</p>
                </div>
              )}
              <AnimatePresence initial={false}>
                {messages.map((msg) => (
                  <motion.div
                    key={msg.id}
                    className={`${styles.message} ${msg.role === 'user' ? styles.messageUser : styles.messageAssistant}`}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.2 }}
                  >
                    {msg.role === 'assistant' && (
                      <div className={styles.messageAvatar}>
                        <LeafIconSm />
                      </div>
                    )}
                    <div className={styles.messageBubble}>
                      <p className={styles.messageContent}>{msg.content}</p>
                      <span className={styles.messageTime}>{formatTime(msg.created_at)}</span>
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>

              {isTyping && (
                <motion.div
                  className={`${styles.message} ${styles.messageAssistant}`}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  <div className={styles.messageAvatar}><LeafIconSm /></div>
                  <div className={`${styles.messageBubble} ${styles.typingBubble}`}>
                    <div className={styles.typingDots}>
                      <span /><span /><span />
                    </div>
                  </div>
                </motion.div>
              )}

              <div ref={messagesEndRef} />
            </div>

            <div className={styles.inputArea}>
              <textarea
                ref={inputRef}
                className={styles.messageInput}
                placeholder="Ask about meals, recipes, or your glucose levels..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                rows={1}
              />
              <button
                className={styles.sendBtn}
                onClick={handleSend}
                disabled={!input.trim() || sendMutation.isPending}
                aria-label="Send message"
              >
                <SendIcon />
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

const SUGGESTIONS = [
  'What can I cook with what I have in my fridge?',
  'My glucose is high — suggest a low-GI dinner',
  'Give me a diabetes-friendly breakfast idea',
  'How many carbs should I aim for per meal?',
]

function LeafIcon() {
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10z"/>
      <path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>
    </svg>
  )
}

function LeafIconSm() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10z"/>
      <path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>
    </svg>
  )
}

function SendIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
    </svg>
  )
}
