import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useAuthStore } from '@/lib/authStore'
import { api } from '@/lib/api'
import { glucoseStatus, glucoseStatusLabel, formatDateTime, capitalise } from '@/lib/utils'
import type { GlucoseReading, MealLog, UserProfile } from '@/types'
import PageHeader from '@/components/layout/PageHeader'
import StatCard from '@/components/ui/StatCard'
import Card from '@/components/ui/Card'
import Badge from '@/components/ui/Badge'
import Button from '@/components/ui/Button'
import styles from './DashboardPage.module.css'
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { format } from 'date-fns'

const GLUCOSE_BADGE: Record<string, 'success' | 'warning' | 'danger' | 'info'> = {
  normal: 'success',
  elevated: 'warning',
  high: 'danger',
  low: 'info',
}

export default function DashboardPage() {
  const token = useAuthStore((s) => s.accessToken)

  const { data: profile } = useQuery<UserProfile>({
    queryKey: ['profile'],
    queryFn: () => api.get('/profile', token ?? undefined),
    enabled: !!token,
  })

  const { data: glucoseReadings } = useQuery<GlucoseReading[]>({
    queryKey: ['glucose'],
    queryFn: () => api.get('/glucose?limit=14', token ?? undefined),
    enabled: !!token,
  })

  const { data: meals } = useQuery<MealLog[]>({
    queryKey: ['meals'],
    queryFn: () => api.get('/meals?limit=5', token ?? undefined),
    enabled: !!token,
  })

  const latestGlucose = glucoseReadings?.[0]
  const avgGlucose = glucoseReadings?.length
    ? (glucoseReadings.reduce((s, r) => s + r.reading_mmol, 0) / glucoseReadings.length).toFixed(1)
    : null

  const todayCarbs = meals
    ?.filter((m) => {
      const today = new Date().toDateString()
      return new Date(m.eaten_at).toDateString() === today
    })
    .reduce((s, m) => s + (m.carbs_grams ?? 0), 0)
    .toFixed(0)

  const chartData = [...(glucoseReadings ?? [])]
    .reverse()
    .map((r) => ({
      time: format(new Date(r.recorded_at), 'MMM d'),
      value: r.reading_mmol,
    }))

  const greeting = () => {
    const h = new Date().getHours()
    if (h < 12) return 'Good morning'
    if (h < 17) return 'Good afternoon'
    return 'Good evening'
  }

  return (
    <div>
      <PageHeader
        title={`${greeting()}${profile?.first_name ? `, ${profile.first_name}` : ''}`}
        subtitle="Here is your health overview for today"
      />

      <div className={styles.statsGrid}>
        <StatCard
          label="Latest glucose"
          value={latestGlucose?.reading_mmol.toFixed(1) ?? '--'}
          unit="mmol/L"
          accent={latestGlucose ? (glucoseStatus(latestGlucose.reading_mmol) === 'normal' ? 'teal' : glucoseStatus(latestGlucose.reading_mmol) === 'elevated' ? 'warning' : glucoseStatus(latestGlucose.reading_mmol) === 'high' ? 'danger' : 'teal') : 'neutral'}
          trendValue={latestGlucose ? glucoseStatusLabel(latestGlucose.reading_mmol) : undefined}
          delay={0}
          icon={<ActivityIcon />}
        />
        <StatCard
          label="14-day average"
          value={avgGlucose ?? '--'}
          unit="mmol/L"
          accent="teal"
          delay={0.05}
          icon={<TrendIcon />}
        />
        <StatCard
          label="Carbs today"
          value={todayCarbs ?? '--'}
          unit="g"
          accent={Number(todayCarbs) > 130 ? 'warning' : 'teal'}
          delay={0.1}
          icon={<UtensilsIcon />}
        />
        <StatCard
          label="Meals logged"
          value={meals?.length ?? 0}
          accent="neutral"
          delay={0.15}
          icon={<ListIcon />}
        />
      </div>

      <div className={styles.mainGrid}>
        <Card elevated className={styles.chartCard}>
          <Card.Header
            title="Glucose trend"
            subtitle="Last 14 readings"
            action={
              <Button variant="ghost" size="sm">
                <Link to="/glucose">View all</Link>
              </Button>
            }
          />
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={chartData} margin={{ top: 4, right: 4, left: -24, bottom: 0 }}>
                <defs>
                  <linearGradient id="glucoseGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#6b9080" stopOpacity={0.2} />
                    <stop offset="95%" stopColor="#6b9080" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="time" tick={{ fontSize: 11, fill: '#8aaa9c' }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 11, fill: '#8aaa9c' }} axisLine={false} tickLine={false} domain={[3, 'auto']} />
                <Tooltip
                  contentStyle={{
                    background: 'white',
                    border: '1px solid #cce3de',
                    borderRadius: '8px',
                    fontSize: '13px',
                    boxShadow: '0 4px 12px rgba(42,56,48,0.1)',
                  }}
                  formatter={(v: number) => [`${v} mmol/L`, 'Glucose']}
                />
                <Area
                  type="monotone"
                  dataKey="value"
                  stroke="#6b9080"
                  strokeWidth={2}
                  fill="url(#glucoseGradient)"
                  dot={{ r: 3, fill: '#6b9080', strokeWidth: 0 }}
                  activeDot={{ r: 5 }}
                />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className={styles.noData}>No readings yet. Log your first glucose reading.</div>
          )}
        </Card>

        <Card elevated className={styles.mealsCard}>
          <Card.Header
            title="Recent meals"
            action={
              <Button variant="ghost" size="sm">
                <Link to="/meals">View all</Link>
              </Button>
            }
          />
          {meals && meals.length > 0 ? (
            <ul className={styles.mealList}>
              {meals.map((meal, i) => (
                <motion.li
                  key={meal.id}
                  className={styles.mealItem}
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.05 }}
                >
                  <div className={styles.mealInfo}>
                    <span className={styles.mealName}>{meal.meal_name}</span>
                    <span className={styles.mealTime}>{formatDateTime(meal.eaten_at)}</span>
                  </div>
                  <div className={styles.mealMeta}>
                    {meal.meal_type && (
                      <Badge variant="muted">{capitalise(meal.meal_type)}</Badge>
                    )}
                    <span className={styles.mealCarbs}>{meal.carbs_grams}g carbs</span>
                  </div>
                </motion.li>
              ))}
            </ul>
          ) : (
            <div className={styles.noData}>No meals logged yet.</div>
          )}
        </Card>
      </div>

      {latestGlucose && glucoseStatus(latestGlucose.reading_mmol) !== 'normal' && (
        <motion.div
          className={styles.insightBanner}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <div className={styles.insightIcon}>
            <LightbulbIcon />
          </div>
          <div className={styles.insightText}>
            <strong>Glucose insight</strong>
            <p>
              Your latest reading of {latestGlucose.reading_mmol} mmol/L is{' '}
              {glucoseStatus(latestGlucose.reading_mmol)}. Consider asking TasteMatch for
              low-GI meal suggestions.
            </p>
          </div>
          <Button variant="secondary" size="sm">
            <Link to="/chat">Ask TasteMatch</Link>
          </Button>
        </motion.div>
      )}
    </div>
  )
}

function ActivityIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12" /></svg>
}
function TrendIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18" /><polyline points="17 6 23 6 23 12" /></svg>
}
function UtensilsIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round"><path d="M3 2v7c0 1.1.9 2 2 2h4a2 2 0 0 0 2-2V2" /><path d="M7 2v20" /><path d="M21 15V2a5 5 0 0 0-5 5v6c0 1.1.9 2 2 2h3Zm0 0v7" /></svg>
}
function ListIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round"><line x1="8" y1="6" x2="21" y2="6" /><line x1="8" y1="12" x2="21" y2="12" /><line x1="8" y1="18" x2="21" y2="18" /><line x1="3" y1="6" x2="3.01" y2="6" /><line x1="3" y1="12" x2="3.01" y2="12" /><line x1="3" y1="18" x2="3.01" y2="18" /></svg>
}
function LightbulbIcon() {
  return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round"><line x1="9" y1="18" x2="15" y2="18" /><line x1="10" y1="22" x2="14" y2="22" /><path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8 6 6 0 0 0 6 8c0 1 .23 2.23 1.5 3.5A4.61 4.61 0 0 1 8.91 14" /></svg>
}
