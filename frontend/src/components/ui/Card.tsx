import { type HTMLAttributes } from 'react'
import { cn } from '@/lib/utils'
import styles from './Card.module.css'

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  elevated?: boolean
  padded?: boolean
}

function Card({ elevated = false, padded = true, className, children, ...props }: CardProps) {
  return (
    <div
      className={cn(
        styles.card,
        elevated && styles.elevated,
        padded && styles.padded,
        className
      )}
      {...props}
    >
      {children}
    </div>
  )
}

interface CardHeaderProps extends HTMLAttributes<HTMLDivElement> {
  title: string
  subtitle?: string
  action?: React.ReactNode
}

function CardHeader({ title, subtitle, action, className, ...props }: CardHeaderProps) {
  return (
    <div className={cn(styles.header, className)} {...props}>
      <div className={styles.headerText}>
        <h3 className={styles.title}>{title}</h3>
        {subtitle && <p className={styles.subtitle}>{subtitle}</p>}
      </div>
      {action && <div className={styles.action}>{action}</div>}
    </div>
  )
}

Card.Header = CardHeader

export default Card
