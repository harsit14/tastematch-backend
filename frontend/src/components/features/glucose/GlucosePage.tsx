import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { useAuthStore } from '@/lib/authStore'
import { api } from '@/lib/api'
import { glucoseStatus, glucoseStatusLabel, formatDateTime } from '@/lib/utils'
import type { GlucoseReading } from '@/types'
import PageHeader from '@/components/layout/PageHeader'
import Card from '@/components/ui/Card'
import Button from '@/components/ui/Button'
import Badge from '@/components/ui/Badge'
import Input from '@/components/ui/Input'
import Modal from '@/components/ui/Modal'
import EmptyState from '@/components/ui/EmptyState'
import { useToast } from '@/components/ui/ToastProvider'
import styles from './GlucosePage.module.css'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { format } from 'date-fns'

const CONTEXT_OPTIONS = ['fasting', 'before_meal', 'after_meal', 'bedtime', 'other']
const BADGE_MAP: Record<string, 'info' | 'success' | 'warning' | 'danger'> = {
  low: 'info', normal: 'success', elevated: 'warning', high: 'danger',
}

export default function GlucosePage() {
  const token = useAuthStore((s) => s.accessToken)
  const qc = useQueryClient()
  const { toast } = useToast()
  const [modalOpen, setModalOpen] = useState(false)
  const [form, setForm] = useState({ reading_mmol: '', reading_context: 'before_meal', notes: '' })

  const { data: readings, isLoading } = useQuery<GlucoseReading[]>({
    queryKey: ['glucose'],
    queryFn: () => api.get('/glucose?limit=50', token ?? undefined),
    enabled: !!token,
  })

  const logMutation = useMutation({
    mutationFn: (body: object) => api.post('/glucose', body, token ?? undefined),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['glucose'] })
      setModalOpen(false)
      setForm({ reading_mmol: '', reading_context: 'before_meal', notes: '' })
      toast('Glucose reading saved', 'success')
    },
    onError: () => toast('Failed to save reading', 'error'),
  })

  const handleLog = () => {
    if (!form.reading_mmol) return
    logMutation.mutate({
      reading_mmol: parseFloat(form.reading_mmol),
      reading_context: form.reading_context,
      notes: form.notes || undefined,
    })
  }

  const chartData = [...(readings ?? [])]
    .reverse()
    .slice(-20)
    .map((r) => ({
      time: format(new Date(r.recorded_at), 'MMM d HH:mm'),
      value: r.reading_mmol,
      context: r.reading_context,
    }))

  const inRange = readings?.filter((r) => glucoseStatus(r.reading_mmol) === 'normal').length ?? 0
  const pct = readings?.length ? Math.round((inRange / readings.length) * 100) : 0

  return (
    <div>
      <PageHeader
        title="Glucose"
        subtitle="Track and visualise your blood sugar over time"
        action={
          <Button onClick={() => setModalOpen(true)}>
            Log reading
          </Button>
        }
      />

      {readings && readings.length > 0 && (
        <div className={styles.summaryRow}>
          <div className={styles.summaryItem}>
            <span className={styles.summaryValue}>{readings[0].reading_mmol.toFixed(1)}</span>
            <span className={styles.summaryLabel}>Latest (mmol/L)</span>
          </div>
          <div className={styles.summaryDivider} />
          <div className={styles.summaryItem}>
            <span className={styles.summaryValue}>
              {(readings.reduce((s, r) => s + r.reading_mmol, 0) / readings.length).toFixed(1)}
            </span>
            <span className={styles.summaryLabel}>Average</span>
          </div>
          <div className={styles.summaryDivider} />
          <div className={styles.summaryItem}>
            <span className={styles.summaryValue}>{pct}%</span>
            <span className={styles.summaryLabel}>In range</span>
          </div>
          <div className={styles.summaryDivider} />
          <div className={styles.summaryItem}>
            <span className={styles.summaryValue}>{readings.length}</span>
            <span className={styles.summaryLabel}>Total readings</span>
          </div>
        </div>
      )}

      {chartData.length > 1 && (
        <Card elevated className={styles.chartCard}>
          <Card.Header title="Blood glucose over time" subtitle="Last 20 readings" />
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={chartData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="lineGrad" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#6b9080" />
                  <stop offset="100%" stopColor="#a4c3b2" />
                </linearGradient>
              </defs>
              <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#8aaa9c' }} axisLine={false} tickLine={false} interval="preserveStartEnd" />
              <YAxis tick={{ fontSize: 10, fill: '#8aaa9c' }} axisLine={false} tickLine={false} domain={[3, 'auto']} />
              <ReferenceLine y={4} stroke="#93c5fd" strokeDasharray="4 4" strokeWidth={1} />
              <ReferenceLine y={7.8} stroke="#fcd34d" strokeDasharray="4 4" strokeWidth={1} />
              <Tooltip
                contentStyle={{ background: 'white', border: '1px solid #cce3de', borderRadius: '8px', fontSize: '12px' }}
                formatter={(v: number) => [`${v} mmol/L`]}
              />
              <Line type="monotone" dataKey="value" stroke="url(#lineGrad)" strokeWidth={2.5} dot={{ r: 3.5, fill: '#6b9080', strokeWidth: 0 }} activeDot={{ r: 5 }} />
            </LineChart>
          </ResponsiveContainer>
          <div className={styles.chartLegend}>
            <span className={styles.legendItem} style={{ color: '#93c5fd' }}>-- Low threshold (4.0)</span>
            <span className={styles.legendItem} style={{ color: '#fcd34d' }}>-- High threshold (7.8)</span>
          </div>
        </Card>
      )}

      <Card elevated>
        <Card.Header title="Reading history" />
        {isLoading ? (
          <div className={styles.loading}>Loading readings...</div>
        ) : readings && readings.length > 0 ? (
          <ul className={styles.list}>
            {readings.map((r, i) => {
              const status = glucoseStatus(r.reading_mmol)
              return (
                <motion.li
                  key={r.id}
                  className={styles.item}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.02 }}
                >
                  <div className={styles.itemLeft}>
                    <span className={`${styles.dot} ${styles[`dot-${status}`]}`} />
                    <div>
                      <span className={styles.itemValue}>{r.reading_mmol.toFixed(1)} mmol/L</span>
                      <span className={styles.itemTime}>{formatDateTime(r.recorded_at)}</span>
                    </div>
                  </div>
                  <div className={styles.itemRight}>
                    {r.reading_context && (
                      <Badge variant="muted">{r.reading_context.replace('_', ' ')}</Badge>
                    )}
                    <Badge variant={BADGE_MAP[status]}>{glucoseStatusLabel(r.reading_mmol)}</Badge>
                    {r.notes && <span className={styles.itemNotes}>{r.notes}</span>}
                  </div>
                </motion.li>
              )
            })}
          </ul>
        ) : (
          <EmptyState title="No readings logged" description="Start tracking your blood glucose by logging your first reading." />
        )}
      </Card>

      <Modal open={modalOpen} onClose={() => setModalOpen(false)} title="Log glucose reading">
        <div className={styles.modalForm}>
          <Input
            label="Blood glucose (mmol/L)"
            type="number"
            step="0.1"
            min="1"
            max="30"
            placeholder="e.g. 6.2"
            value={form.reading_mmol}
            onChange={(e) => setForm((p) => ({ ...p, reading_mmol: e.target.value }))}
          />
          <div className={styles.selectWrapper}>
            <label className={styles.selectLabel}>Context</label>
            <select
              className={styles.select}
              value={form.reading_context}
              onChange={(e) => setForm((p) => ({ ...p, reading_context: e.target.value }))}
            >
              {CONTEXT_OPTIONS.map((o) => (
                <option key={o} value={o}>{o.replace('_', ' ')}</option>
              ))}
            </select>
          </div>
          <Input
            label="Notes (optional)"
            placeholder="Any relevant notes..."
            value={form.notes}
            onChange={(e) => setForm((p) => ({ ...p, notes: e.target.value }))}
          />
          {form.reading_mmol && (
            <p className={styles.readingPreview}>
              Status: <strong>{glucoseStatusLabel(parseFloat(form.reading_mmol))}</strong>
            </p>
          )}
          <div className={styles.modalActions}>
            <Button variant="secondary" onClick={() => setModalOpen(false)}>Cancel</Button>
            <Button onClick={handleLog} loading={logMutation.isPending}>Save reading</Button>
          </div>
        </div>
      </Modal>
    </div>
  )
}
