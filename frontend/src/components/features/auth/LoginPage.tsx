import { useState, type FormEvent } from 'react'
import { Link, useNavigate, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import { api, ApiError } from '@/lib/api'
import { useAuthStore } from '@/lib/authStore'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import styles from './AuthPages.module.css'

export default function LoginPage() {
  const navigate = useNavigate()
  const location = useLocation()
  const setAuth = useAuthStore((s) => s.setAuth)

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const infoMessage = (location.state as { message?: string } | null)?.message

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    try {
      const data = await api.post<{ user_id: string; access_token: string; refresh_token: string; expires_in: number }>(
        '/auth/login',
        { email, password }
      )
      setAuth({ id: data.user_id, email }, data.access_token, data.refresh_token, data.expires_in ?? 3600)
      navigate('/dashboard')
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={styles.page}>
      <div className={styles.decorPanel}>
        <div className={styles.decorContent}>
          <div className={styles.decorQuote}>
            <p>"Food is not just fuel. It is information. It talks to your DNA."</p>
            <span>— Mark Hyman</span>
          </div>
          <div className={styles.decorOrbs} aria-hidden="true">
            <div className={styles.orb1} />
            <div className={styles.orb2} />
            <div className={styles.orb3} />
          </div>
        </div>
      </div>

      <div className={styles.formPanel}>
        <motion.div
          className={styles.formContainer}
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, ease: 'easeOut' }}
        >
          <div className={styles.logoRow}>
            <div className={styles.logo}>
              <LeafIcon />
            </div>
            <span className={styles.logoLabel}>TasteMatch</span>
          </div>

          <div className={styles.formHead}>
            <h1 className={styles.formTitle}>Welcome back</h1>
            <p className={styles.formSubtitle}>Sign in to your nutrition dashboard</p>
          </div>

          <form className={styles.form} onSubmit={handleSubmit}>
            <Input
              label="Email address"
              type="email"
              placeholder="you@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              autoComplete="email"
            />
            <Input
              label="Password"
              type="password"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              autoComplete="current-password"
            />

            {infoMessage && <p className={styles.formInfo}>{infoMessage}</p>}
            {error && <p className={styles.formError}>{error}</p>}

            <Button type="submit" fullWidth loading={loading} size="lg">
              Sign in
            </Button>
          </form>

          <p className={styles.switchText}>
            No account yet?{' '}
            <Link to="/signup" className={styles.switchLink}>
              Create one
            </Link>
          </p>
        </motion.div>
      </div>
    </div>
  )
}

function LeafIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
      <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10z"/>
      <path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>
    </svg>
  )
}
