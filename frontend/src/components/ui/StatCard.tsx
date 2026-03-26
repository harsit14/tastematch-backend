import { type ReactNode } from 'react'
import { motion } from 'framer-motion'
import styles from './StatCard.module.css'
import { cn } from '@/lib/utils'

interface StatCardProps {
  label: string
  value: string | number
  unit?: string
  trend?: 'up' | 'down' | 'neutral'
  trendValue?: string
  accent?: 'teal' | 'warning' | 'danger' | 'neutral'
  icon?: ReactNode
  delay?: number
}

export default function StatCard({
  label,
  value,
  unit,
  trend,
  trendValue,
  accent = 'teal',
  icon,
  delay = 0,
}: StatCardProps) {
  return (
    <motion.div
      className={cn(styles.card, styles[accent])}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay, ease: 'easeOut' }}
    >
      <div className={styles.top}>
        <span className={styles.label}>{label}</span>
        {icon && <span className={styles.icon}>{icon}</span>}
      </div>
      <div className={styles.valueRow}>
        <span className={styles.value}>{value}</span>
        {unit && <span className={styles.unit}>{unit}</span>}
      </div>
      {trendValue && (
        <div className={cn(styles.trend, trend && styles[`trend-${trend}`])}>
          {trend === 'up' && <ArrowUpIcon />}
          {trend === 'down' && <ArrowDownIcon />}
          <span>{trendValue}</span>
        </div>
      )}
    </motion.div>
  )
}

function ArrowUpIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="18 15 12 9 6 15" />
    </svg>
  )
}

function ArrowDownIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="6 9 12 15 18 9" />
    </svg>
  )
}
