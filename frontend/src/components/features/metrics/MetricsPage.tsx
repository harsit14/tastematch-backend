import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { useAuthStore } from '@/lib/authStore'
import { api } from '@/lib/api'
import { bmiCategory, formatDate } from '@/lib/utils'
import type { BodyMetric } from '@/types'
import PageHeader from '@/components/layout/PageHeader'
import Card from '@/components/ui/Card'
import Button from '@/components/ui/Button'
import Badge from '@/components/ui/Badge'
import Input from '@/components/ui/Input'
import Modal from '@/components/ui/Modal'
import EmptyState from '@/components/ui/EmptyState'
import { useToast } from '@/components/ui/ToastProvider'
import styles from './MetricsPage.module.css'
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { format } from 'date-fns'

export default function MetricsPage() {
  const token = useAuthStore((s) => s.accessToken)
  const qc = useQueryClient()
  const { toast } = useToast()
  const [modalOpen, setModalOpen] = useState(false)
  const [form, setForm] = useState({ weight_kg: '', height_cm: '' })

  const { data: metrics, isLoading } = useQuery<BodyMetric[]>({
    queryKey: ['body-metrics'],
    queryFn: () => api.get('/body-metrics?limit=30', token ?? undefined),
    enabled: !!token,
  })

  const logMutation = useMutation({
    mutationFn: (body: object) => api.post('/body-metrics', body, token ?? undefined),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['body-metrics'] })
      setModalOpen(false)
      setForm({ weight_kg: '', height_cm: '' })
      toast('Measurement saved', 'success')
    },
    onError: () => toast('Failed to save measurement', 'error'),
  })

  const handleLog = () => {
    if (!form.weight_kg) return
    logMutation.mutate({
      weight_kg: parseFloat(form.weight_kg),
      height_cm: form.height_cm ? parseFloat(form.height_cm) : (metrics?.[0]?.height_cm ?? 170),
    })
  }

  const set = (f: string) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm((p) => ({ ...p, [f]: e.target.value }))

  const latest = metrics?.[0]
  const weightData = [...(metrics ?? [])]
    .reverse()
    .map((m) => ({ date: format(new Date(m.recorded_at), 'MMM d'), weight: m.weight_kg, bmi: m.bmi }))

  const bmiVariant = (bmi: number): 'success' | 'warning' | 'danger' | 'info' => {
    if (bmi < 18.5) return 'info'
    if (bmi < 25) return 'success'
    if (bmi < 30) return 'warning'
    return 'danger'
  }

  return (
    <div>
      <PageHeader
        title="Body Metrics"
        subtitle="Track weight and BMI trends over time"
        action={
          <Button onClick={() => {
            setModalOpen(true)
            if (metrics && metrics.length > 0) {
              setForm(p => ({ ...p, height_cm: String(metrics[0].height_cm) }))
            }
          }}>Log measurement</Button>
        }
      />

      {latest && (
        <div className={styles.latestRow}>
          <div className={styles.latestCard}>
            <span className={styles.latestLabel}>Current weight</span>
            <div className={styles.latestValue}>
              <span className={styles.latestNum}>{latest.weight_kg}</span>
              <span className={styles.latestUnit}>kg</span>
            </div>
          </div>
          <div className={styles.latestCard}>
            <span className={styles.latestLabel}>Height</span>
            <div className={styles.latestValue}>
              <span className={styles.latestNum}>{latest.height_cm}</span>
              <span className={styles.latestUnit}>cm</span>
            </div>
          </div>
          <div className={styles.latestCard}>
            <span className={styles.latestLabel}>BMI</span>
            <div className={styles.latestValue}>
              <span className={styles.latestNum}>{latest.bmi?.toFixed(1)}</span>
            </div>
            <Badge variant={bmiVariant(latest.bmi)}>{bmiCategory(latest.bmi)}</Badge>
          </div>
          <div className={styles.latestCard}>
            <span className={styles.latestLabel}>Measurements</span>
            <div className={styles.latestValue}>
              <span className={styles.latestNum}>{metrics?.length}</span>
            </div>
          </div>
        </div>
      )}

      {weightData.length > 1 && (
        <Card elevated className={styles.chartCard}>
          <Card.Header title="Weight trend" subtitle="All recorded measurements" />
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={weightData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="weightGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#6b9080" stopOpacity={0.18} />
                  <stop offset="95%" stopColor="#6b9080" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#8aaa9c' }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 11, fill: '#8aaa9c' }} axisLine={false} tickLine={false} domain={['auto', 'auto']} />
              <Tooltip
                contentStyle={{ background: 'white', border: '1px solid #cce3de', borderRadius: '8px', fontSize: '12px' }}
                formatter={(v: number) => [`${v} kg`, 'Weight']}
              />
              <Area type="monotone" dataKey="weight" stroke="#6b9080" strokeWidth={2} fill="url(#weightGrad)" dot={{ r: 3, fill: '#6b9080', strokeWidth: 0 }} />
            </AreaChart>
          </ResponsiveContainer>
        </Card>
      )}

      <Card elevated>
        <Card.Header title="Measurement history" />
        {isLoading ? (
          <div className={styles.loading}>Loading metrics...</div>
        ) : metrics && metrics.length > 0 ? (
          <ul className={styles.list}>
            {metrics.map((m, i) => (
              <motion.li
                key={m.id}
                className={styles.item}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.03 }}
              >
                <span className={styles.itemDate}>{formatDate(m.recorded_at)}</span>
                <div className={styles.itemStats}>
                  <span className={styles.itemStat}><strong>{m.weight_kg}</strong> kg</span>
                  <span className={styles.itemStat}><strong>{m.height_cm}</strong> cm</span>
                  <span className={styles.itemStat}>BMI <strong>{m.bmi?.toFixed(1)}</strong></span>
                  <Badge variant={bmiVariant(m.bmi)}>{bmiCategory(m.bmi)}</Badge>
                </div>
              </motion.li>
            ))}
          </ul>
        ) : (
          <EmptyState title="No measurements logged" description="Log your first weight and height to start tracking your BMI trend." icon={<TrendIcon />} />
        )}
      </Card>

      <Modal open={modalOpen} onClose={() => setModalOpen(false)} title="Log measurement" size="sm">
        <div className={styles.modalForm}>
          <Input label="Weight (kg)" type="number" step="0.1" min="20" max="300" placeholder="e.g. 74.5" value={form.weight_kg} onChange={set('weight_kg')} />
          <Input label="Height (cm)" type="number" step="0.5" min="100" max="250" placeholder="e.g. 170" value={form.height_cm} onChange={set('height_cm')} />
          {form.weight_kg && form.height_cm && (
            <div className={styles.bmiPreview}>
              BMI preview:{' '}
              <strong>
                {(parseFloat(form.weight_kg) / Math.pow(parseFloat(form.height_cm) / 100, 2)).toFixed(1)}
              </strong>
            </div>
          )}
          {logMutation.isError && <p className={styles.error}>Failed to log measurement.</p>}
          <div className={styles.modalActions}>
            <Button variant="secondary" onClick={() => setModalOpen(false)}>Cancel</Button>
            <Button onClick={handleLog} loading={logMutation.isPending}>Save</Button>
          </div>
        </div>
      </Modal>
    </div>
  )
}

function TrendIcon() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>
    </svg>
  )
}
