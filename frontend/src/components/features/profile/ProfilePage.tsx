import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useAuthStore } from '@/lib/authStore'
import { api } from '@/lib/api'
import { capitalise, formatDate } from '@/lib/utils'
import type { UserProfile } from '@/types'
import PageHeader from '@/components/layout/PageHeader'
import Card from '@/components/ui/Card'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import Badge from '@/components/ui/Badge'
import { useToast } from '@/components/ui/ToastProvider'
import styles from './ProfilePage.module.css'

const DIABETES_OPTIONS = ['type1', 'type2', 'prediabetes', 'gestational', 'other', 'none']

const DIETARY_OPTIONS = [
  'vegetarian', 'vegan', 'gluten-free', 'dairy-free',
  'low-carb', 'keto', 'halal', 'kosher', 'paleo',
]

const ALLERGY_OPTIONS = [
  'nuts', 'peanuts', 'shellfish', 'fish', 'eggs',
  'milk', 'soy', 'wheat', 'sesame',
]

export default function ProfilePage() {
  const token = useAuthStore((s) => s.accessToken)
  const qc = useQueryClient()
  const { toast } = useToast()
  const [editing, setEditing] = useState(false)
  const [form, setForm] = useState({
    first_name: '',
    last_name: '',
    date_of_birth: '',
    diabetes_type: 'none',
    dietary_preferences: [] as string[],
    allergies: [] as string[],
  })

  const { data: profile, isLoading } = useQuery<UserProfile>({
    queryKey: ['profile'],
    queryFn: () => api.get('/profile', token ?? undefined),
    enabled: !!token,
  })

  useEffect(() => {
    if (profile) {
      setForm({
        first_name: profile.first_name ?? '',
        last_name: profile.last_name ?? '',
        date_of_birth: profile.date_of_birth ?? '',
        diabetes_type: profile.diabetes_type ?? 'none',
        dietary_preferences: profile.dietary_preferences ?? [],
        allergies: profile.allergies ?? [],
      })
    }
  }, [profile])

  const updateMutation = useMutation({
    mutationFn: (body: object) => api.put('/profile', body, token ?? undefined),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['profile'] })
      setEditing(false)
      toast('Profile updated', 'success')
    },
    onError: () => toast('Failed to update profile', 'error'),
  })

  const handleSave = () => updateMutation.mutate(form)

  const toggleArrayItem = (arr: string[], item: string): string[] =>
    arr.includes(item) ? arr.filter((i) => i !== item) : [...arr, item]

  if (isLoading) {
    return (
      <div>
        <PageHeader title="Profile" />
        <div className={styles.loading}>Loading profile...</div>
      </div>
    )
  }

  return (
    <div>
      <PageHeader
        title="Profile"
        subtitle="Manage your health profile and dietary preferences"
        action={
          editing ? (
            <div className={styles.editActions}>
              <Button variant="secondary" onClick={() => setEditing(false)}>Discard</Button>
              <Button onClick={handleSave} loading={updateMutation.isPending}>Save changes</Button>
            </div>
          ) : (
            <Button variant="secondary" onClick={() => setEditing(true)}>Edit profile</Button>
          )
        }
      />

      <div className={styles.grid}>
        <Card elevated>
          <Card.Header title="Personal details" />
          {editing ? (
            <div className={styles.editForm}>
              <div className={styles.row}>
                <Input label="First name" value={form.first_name} onChange={(e) => setForm((p) => ({ ...p, first_name: e.target.value }))} />
                <Input label="Last name" value={form.last_name} onChange={(e) => setForm((p) => ({ ...p, last_name: e.target.value }))} />
              </div>
              <Input label="Date of birth" type="date" value={form.date_of_birth} onChange={(e) => setForm((p) => ({ ...p, date_of_birth: e.target.value }))} />
              <div className={styles.selectWrapper}>
                <label className={styles.selectLabel}>Diabetes type</label>
                <select className={styles.select} value={form.diabetes_type} onChange={(e) => setForm((p) => ({ ...p, diabetes_type: e.target.value }))}>
                  {DIABETES_OPTIONS.map((o) => <option key={o} value={o}>{capitalise(o)}</option>)}
                </select>
              </div>
            </div>
          ) : (
            <div className={styles.detailList}>
              <DetailRow label="Name" value={`${profile?.first_name ?? ''} ${profile?.last_name ?? ''}`.trim() || '--'} />
              <DetailRow label="Date of birth" value={profile?.date_of_birth ? formatDate(profile.date_of_birth) : '--'} />
              <DetailRow label="Age" value={profile?.age ? `${profile.age} years` : '--'} />
              <DetailRow
                label="Diabetes type"
                value={
                  <Badge variant={profile?.diabetes_type === 'none' ? 'muted' : 'default'}>
                    {capitalise(profile?.diabetes_type ?? 'none')}
                  </Badge>
                }
              />
            </div>
          )}
        </Card>

        <Card elevated>
          <Card.Header title="Dietary preferences" />
          {editing ? (
            <div className={styles.tagGrid}>
              {DIETARY_OPTIONS.map((opt) => (
                <button
                  key={opt}
                  className={`${styles.tagToggle} ${form.dietary_preferences.includes(opt) ? styles.tagToggleActive : ''}`}
                  onClick={() => setForm((p) => ({ ...p, dietary_preferences: toggleArrayItem(p.dietary_preferences, opt) }))}
                >
                  {capitalise(opt)}
                </button>
              ))}
            </div>
          ) : (
            <div className={styles.tagDisplay}>
              {profile?.dietary_preferences?.length ? (
                profile.dietary_preferences.map((p) => <Badge key={p} variant="success">{capitalise(p)}</Badge>)
              ) : (
                <span className={styles.emptyText}>No preferences set</span>
              )}
            </div>
          )}
        </Card>

        <Card elevated>
          <Card.Header title="Allergies" subtitle="Foods to always avoid in suggestions" />
          {editing ? (
            <div className={styles.tagGrid}>
              {ALLERGY_OPTIONS.map((opt) => (
                <button
                  key={opt}
                  className={`${styles.tagToggle} ${styles.tagToggleDanger} ${form.allergies.includes(opt) ? styles.tagToggleDangerActive : ''}`}
                  onClick={() => setForm((p) => ({ ...p, allergies: toggleArrayItem(p.allergies, opt) }))}
                >
                  {capitalise(opt)}
                </button>
              ))}
            </div>
          ) : (
            <div className={styles.tagDisplay}>
              {profile?.allergies?.length ? (
                profile.allergies.map((a) => <Badge key={a} variant="danger">{capitalise(a)}</Badge>)
              ) : (
                <span className={styles.emptyText}>No allergies recorded</span>
              )}
            </div>
          )}
        </Card>
      </div>

      {updateMutation.isError && (
        <p className={styles.error}>Failed to update profile. Please try again.</p>
      )}
    </div>
  )
}

function DetailRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className={styles.detailRow}>
      <span className={styles.detailLabel}>{label}</span>
      <span className={styles.detailValue}>{value}</span>
    </div>
  )
}
