import { useState, type FormEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { api, ApiError } from '@/lib/api'
import { useAuthStore } from '@/lib/authStore'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import styles from './AuthPages.module.css'

const DIABETES_OPTIONS = [
  { value: 'type1', label: 'Type 1' },
  { value: 'type2', label: 'Type 2' },
  { value: 'prediabetes', label: 'Prediabetes' },
  { value: 'gestational', label: 'Gestational' },
  { value: 'other', label: 'Other' },
  { value: 'none', label: 'None' },
]

export default function SignupPage() {
  const navigate = useNavigate()
  const setAuth = useAuthStore((s) => s.setAuth)

  const [form, setForm] = useState({
    email: '',
    password: '',
    first_name: '',
    last_name: '',
    date_of_birth: '',
    diabetes_type: 'none',
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const set = (field: string) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
    setForm((prev) => ({ ...prev, [field]: e.target.value }))

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    try {
      const data = await api.post<{ user_id: string; access_token: string | null; refresh_token?: string; expires_in?: number }>(
        '/auth/signup',
        { ...form }
      )
      if (data.access_token) {
        setAuth({ id: data.user_id, email: form.email }, data.access_token, data.refresh_token ?? '', data.expires_in ?? 3600)
        navigate('/dashboard')
      } else {
        // Supabase email confirmation required — token not returned until confirmed
        navigate('/login', { state: { message: 'Account created! Check your email to confirm, then sign in.' } })
      }
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
            <p>"Let food be thy medicine and medicine be thy food."</p>
            <span>— Hippocrates</span>
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
            <h1 className={styles.formTitle}>Create your account</h1>
            <p className={styles.formSubtitle}>Start your personalised nutrition journey</p>
          </div>

          <form className={styles.form} onSubmit={handleSubmit}>
            <div className={styles.row}>
              <Input
                label="First name"
                placeholder="Jane"
                value={form.first_name}
                onChange={set('first_name')}
                required
              />
              <Input
                label="Last name"
                placeholder="Doe"
                value={form.last_name}
                onChange={set('last_name')}
                required
              />
            </div>

            <Input
              label="Email address"
              type="email"
              placeholder="you@example.com"
              value={form.email}
              onChange={set('email')}
              required
              autoComplete="email"
            />

            <Input
              label="Password"
              type="password"
              placeholder="Minimum 8 characters"
              value={form.password}
              onChange={set('password')}
              required
              minLength={8}
            />

            <Input
              label="Date of birth"
              type="date"
              value={form.date_of_birth}
              onChange={set('date_of_birth')}
              required
            />

            <div className={styles.selectWrapper}>
              <label className={styles.selectLabel} htmlFor="diabetes_type">
                Diabetes type
              </label>
              <select
                id="diabetes_type"
                className={styles.select}
                value={form.diabetes_type}
                onChange={set('diabetes_type')}
              >
                {DIABETES_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>

            {error && <p className={styles.formError}>{error}</p>}

            <Button type="submit" fullWidth loading={loading} size="lg">
              Create account
            </Button>
          </form>

          <p className={styles.switchText}>
            Already have an account?{' '}
            <Link to="/login" className={styles.switchLink}>
              Sign in
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
