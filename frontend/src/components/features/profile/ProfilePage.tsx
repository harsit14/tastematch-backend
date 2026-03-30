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
    glucose_low_target: '',
    glucose_high_target: '',
    carb_target_grams: '',
    hba1c: '',
    hba1c_date: '',
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
        glucose_low_target: profile.glucose_low_target?.toString() ?? '',
        glucose_high_target: profile.glucose_high_target?.toString() ?? '',
        carb_target_grams: profile.carb_target_grams?.toString() ?? '',
        hba1c: profile.hba1c?.toString() ?? '',
        hba1c_date: profile.hba1c_date ?? '',
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

  const handleSave = () => updateMutation.mutate({
    ...form,
    glucose_low_target: form.glucose_low_target ? parseFloat(form.glucose_low_target) : undefined,
    glucose_high_target: form.glucose_high_target ? parseFloat(form.glucose_high_target) : undefined,
    carb_target_grams: form.carb_target_grams ? parseFloat(form.carb_target_grams) : undefined,
    hba1c: form.hba1c ? parseFloat(form.hba1c) : undefined,
    hba1c_date: form.hba1c_date || undefined,
  })

  const toggleArrayItem = (arr: string[], item: string): string[] =>
    arr.includes(item) ? arr.filter((i) => i !== item) : [...arr, item]

  const displayAge = (() => {
    const dob = profile?.date_of_birth
    if (!dob) return null
    const birth = new Date(dob)
    if (isNaN(birth.getTime())) return null
    const today = new Date()
    let age = today.getFullYear() - birth.getFullYear()
    const m = today.getMonth() - birth.getMonth()
    if (m < 0 || (m === 0 && today.getDate() < birth.getDate())) age--
    return age
  })()

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
              <DetailRow label="Age" value={displayAge !== null ? `${displayAge} years` : '--'} />
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
          <Card.Header title="Clinical targets" subtitle="Personalise your glucose and nutrition goals" />
          {editing ? (
            <div className={styles.editForm}>
              <div className={styles.row}>
                <Input label="Low glucose target (mmol/L)" type="number" step="0.1" min="2" max="10"
                  placeholder="e.g. 4.0"
                  value={form.glucose_low_target}
                  onChange={(e) => setForm(p => ({ ...p, glucose_low_target: e.target.value }))} />
                <Input label="High glucose target (mmol/L)" type="number" step="0.1" min="5" max="20"
                  placeholder="e.g. 7.8"
                  value={form.glucose_high_target}
                  onChange={(e) => setForm(p => ({ ...p, glucose_high_target: e.target.value }))} />
              </div>
              <Input label="Daily carb target (g)" type="number" min="0" max="600"
                placeholder="e.g. 130"
                value={form.carb_target_grams}
                onChange={(e) => setForm(p => ({ ...p, carb_target_grams: e.target.value }))} />
              <div className={styles.row}>
                <Input label="Latest HbA1c (%)" type="number" step="0.1" min="3" max="20"
                  placeholder="e.g. 7.2"
                  value={form.hba1c}
                  onChange={(e) => setForm(p => ({ ...p, hba1c: e.target.value }))} />
                <Input label="HbA1c date" type="date"
                  value={form.hba1c_date}
                  onChange={(e) => setForm(p => ({ ...p, hba1c_date: e.target.value }))} />
              </div>
            </div>
          ) : (
            <div className={styles.detailList}>
              <DetailRow label="Glucose target range"
                value={profile?.glucose_low_target || profile?.glucose_high_target
                  ? `${profile?.glucose_low_target ?? 4.0}–${profile?.glucose_high_target ?? 7.8} mmol/L`
                  : '--'} />
              <DetailRow label="Daily carb target"
                value={profile?.carb_target_grams ? `${profile.carb_target_grams}g` : '--'} />
              <DetailRow label="Latest HbA1c"
                value={profile?.hba1c
                  ? `${profile.hba1c}%${profile.hba1c_date ? ` (${formatDate(profile.hba1c_date)})` : ''}`
                  : '--'} />
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
