import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { useAuthStore } from '@/lib/authStore'
import { api } from '@/lib/api'
import { formatDateTime, capitalise } from '@/lib/utils'
import type { MealLog } from '@/types'
import PageHeader from '@/components/layout/PageHeader'
import Card from '@/components/ui/Card'
import Button from '@/components/ui/Button'
import Badge from '@/components/ui/Badge'
import Input from '@/components/ui/Input'
import Modal from '@/components/ui/Modal'
import EmptyState from '@/components/ui/EmptyState'
import { useToast } from '@/components/ui/ToastProvider'
import styles from './MealsPage.module.css'

const MEAL_TYPES = ['breakfast', 'lunch', 'dinner', 'snack']

const MEAL_TYPE_COLORS: Record<string, 'default' | 'success' | 'warning' | 'info' | 'muted'> = {
  breakfast: 'info',
  lunch: 'success',
  dinner: 'default',
  snack: 'warning',
}

export default function MealsPage() {
  const token = useAuthStore((s) => s.accessToken)
  const qc = useQueryClient()
  const { toast } = useToast()
  const [modalOpen, setModalOpen] = useState(false)
  const [filter, setFilter] = useState<string>('all')
  const [form, setForm] = useState({
    meal_name: '',
    carbs_grams: '',
    calories: '',
    meal_type: 'lunch',
    notes: '',
  })

  const { data: meals, isLoading } = useQuery<MealLog[]>({
    queryKey: ['meals'],
    queryFn: () => api.get('/meals?limit=100', token ?? undefined),
    enabled: !!token,
  })

  const logMutation = useMutation({
    mutationFn: (body: object) => api.post('/meals', body, token ?? undefined),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['meals'] })
      setModalOpen(false)
      setForm({ meal_name: '', carbs_grams: '', calories: '', meal_type: 'lunch', notes: '' })
      toast('Meal logged successfully', 'success')
    },
    onError: () => toast('Failed to log meal', 'error'),
  })

  const handleLog = () => {
    if (!form.meal_name || !form.carbs_grams) return
    logMutation.mutate({
      meal_name: form.meal_name,
      carbs_grams: parseFloat(form.carbs_grams),
      calories: form.calories ? parseFloat(form.calories) : undefined,
      meal_type: form.meal_type,
      notes: form.notes || undefined,
    })
  }

  const set = (f: string) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
    setForm((p) => ({ ...p, [f]: e.target.value }))

  const filtered = filter === 'all' ? meals : meals?.filter((m) => m.meal_type === filter)

  const totalCarbs = meals?.reduce((s, m) => s + (m.carbs_grams ?? 0), 0) ?? 0
  const avgCarbs = meals?.length ? (totalCarbs / meals.length).toFixed(0) : '--'
  const totalCalories = meals?.reduce((s, m) => s + (m.calories ?? 0), 0) ?? 0

  return (
    <div>
      <PageHeader
        title="Meals"
        subtitle="Monitor your carbohydrate and calorie intake"
        action={<Button onClick={() => setModalOpen(true)}>Log meal</Button>}
      />

      {meals && meals.length > 0 && (
        <div className={styles.summaryRow}>
          <div className={styles.summaryItem}>
            <span className={styles.summaryValue}>{meals.length}</span>
            <span className={styles.summaryLabel}>Meals logged</span>
          </div>
          <div className={styles.summaryDivider} />
          <div className={styles.summaryItem}>
            <span className={styles.summaryValue}>{avgCarbs}g</span>
            <span className={styles.summaryLabel}>Avg carbs/meal</span>
          </div>
          <div className={styles.summaryDivider} />
          <div className={styles.summaryItem}>
            <span className={styles.summaryValue}>{totalCarbs.toFixed(0)}g</span>
            <span className={styles.summaryLabel}>Total carbs</span>
          </div>
          <div className={styles.summaryDivider} />
          <div className={styles.summaryItem}>
            <span className={styles.summaryValue}>{totalCalories.toFixed(0)}</span>
            <span className={styles.summaryLabel}>Total calories</span>
          </div>
        </div>
      )}

      <Card elevated>
        <div className={styles.filterRow}>
          {['all', ...MEAL_TYPES].map((f) => (
            <button
              key={f}
              className={`${styles.filterBtn} ${filter === f ? styles.filterBtnActive : ''}`}
              onClick={() => setFilter(f)}
            >
              {capitalise(f)}
            </button>
          ))}
        </div>

        {isLoading ? (
          <div className={styles.loading}>Loading meals...</div>
        ) : filtered && filtered.length > 0 ? (
          <div className={styles.mealGrid}>
            {filtered.map((meal, i) => (
              <motion.div
                key={meal.id}
                className={styles.mealCard}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.03 }}
              >
                <div className={styles.mealCardTop}>
                  <div className={styles.mealCardInfo}>
                    <span className={styles.mealCardName}>{meal.meal_name}</span>
                    <span className={styles.mealCardTime}>{formatDateTime(meal.eaten_at)}</span>
                  </div>
                  {meal.meal_type && (
                    <Badge variant={MEAL_TYPE_COLORS[meal.meal_type] ?? 'muted'}>
                      {capitalise(meal.meal_type)}
                    </Badge>
                  )}
                </div>
                <div className={styles.mealCardStats}>
                  <div className={styles.mealCardStat}>
                    <span className={styles.mealCardStatValue}>{meal.carbs_grams}g</span>
                    <span className={styles.mealCardStatLabel}>Carbs</span>
                  </div>
                  {meal.calories != null && (
                    <div className={styles.mealCardStat}>
                      <span className={styles.mealCardStatValue}>{meal.calories}</span>
                      <span className={styles.mealCardStatLabel}>kcal</span>
                    </div>
                  )}
                  {meal.glycemic_load != null && (
                    <div className={styles.mealCardStat}>
                      <span className={styles.mealCardStatValue}>{meal.glycemic_load.toFixed(1)}</span>
                      <span className={styles.mealCardStatLabel}>GL</span>
                    </div>
                  )}
                </div>
                {meal.notes && <p className={styles.mealCardNotes}>{meal.notes}</p>}
              </motion.div>
            ))}
          </div>
        ) : (
          <EmptyState
            title="No meals logged"
            description="Start tracking your meals to gain insights into your carbohydrate intake."
            icon={<UtensilsIcon />}
          />
        )}
      </Card>

      <Modal open={modalOpen} onClose={() => setModalOpen(false)} title="Log a meal">
        <div className={styles.modalForm}>
          <Input
            label="Meal name"
            placeholder="e.g. Grilled salmon with rice"
            value={form.meal_name}
            onChange={set('meal_name')}
          />
          <div className={styles.row}>
            <Input
              label="Carbohydrates (g)"
              type="number"
              min="0"
              placeholder="e.g. 45"
              value={form.carbs_grams}
              onChange={set('carbs_grams')}
            />
            <Input
              label="Calories (optional)"
              type="number"
              min="0"
              placeholder="e.g. 520"
              value={form.calories}
              onChange={set('calories')}
            />
          </div>
          <div className={styles.selectWrapper}>
            <label className={styles.selectLabel}>Meal type</label>
            <select className={styles.select} value={form.meal_type} onChange={set('meal_type')}>
              {MEAL_TYPES.map((t) => (
                <option key={t} value={t}>{capitalise(t)}</option>
              ))}
            </select>
          </div>
          <Input
            label="Notes (optional)"
            placeholder="Any additional notes..."
            value={form.notes}
            onChange={set('notes')}
          />
          {logMutation.isError && (
            <p className={styles.error}>Failed to log meal. Please try again.</p>
          )}
          <div className={styles.modalActions}>
            <Button variant="secondary" onClick={() => setModalOpen(false)}>Cancel</Button>
            <Button onClick={handleLog} loading={logMutation.isPending}>Save meal</Button>
          </div>
        </div>
      </Modal>
    </div>
  )
}

function UtensilsIcon() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 2v7c0 1.1.9 2 2 2h4a2 2 0 0 0 2-2V2"/><path d="M7 2v20"/>
      <path d="M21 15V2a5 5 0 0 0-5 5v6c0 1.1.9 2 2 2h3Zm0 0v7"/>
    </svg>
  )
}
