import { cn } from '@/lib/utils'
import styles from './Skeleton.module.css'

interface SkeletonProps {
  width?: string | number
  height?: string | number
  className?: string
  rounded?: boolean
}

export function Skeleton({ width, height, className, rounded = false }: SkeletonProps) {
  return (
    <div
      className={cn(styles.skeleton, rounded && styles.rounded, className)}
      style={{ width, height }}
      aria-hidden="true"
    />
  )
}

export function SkeletonCard() {
  return (
    <div className={styles.card}>
      <Skeleton height={16} width="60%" className={styles.mb} />
      <Skeleton height={40} width="40%" className={styles.mb} />
      <Skeleton height={12} width="80%" />
    </div>
  )
}

export function SkeletonListItem() {
  return (
    <div className={styles.listItem}>
      <div className={styles.listItemLeft}>
        <Skeleton width={8} height={8} rounded />
        <div>
          <Skeleton height={14} width={160} className={styles.mb} />
          <Skeleton height={11} width={100} />
        </div>
      </div>
      <Skeleton height={20} width={60} rounded />
    </div>
  )
}
