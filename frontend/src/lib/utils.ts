import { clsx, type ClassValue } from 'clsx'

export function cn(...inputs: ClassValue[]) {
  return clsx(inputs)
}

export function formatDate(dateStr: string): string {
  return new Date(dateStr).toLocaleDateString('en-GB', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
  })
}

export function formatTime(dateStr: string): string {
  return new Date(dateStr).toLocaleTimeString('en-GB', {
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function formatDateTime(dateStr: string): string {
  return `${formatDate(dateStr)}, ${formatTime(dateStr)}`
}

export function glucoseStatus(mmol: number): 'low' | 'normal' | 'elevated' | 'high' {
  if (mmol < 4.0) return 'low'
  if (mmol <= 7.8) return 'normal'
  if (mmol <= 11.0) return 'elevated'
  return 'high'
}

export function glucoseStatusLabel(mmol: number): string {
  const status = glucoseStatus(mmol)
  const labels: Record<string, string> = {
    low: 'Low',
    normal: 'In range',
    elevated: 'Elevated',
    high: 'High',
  }
  return labels[status]
}

export function bmiCategory(bmi: number): string {
  if (bmi < 18.5) return 'Underweight'
  if (bmi < 25) return 'Healthy'
  if (bmi < 30) return 'Overweight'
  return 'Obese'
}

export function capitalise(str: string): string {
  return str.charAt(0).toUpperCase() + str.slice(1).replace(/_/g, ' ')
}

export function daysUntilExpiry(dateStr: string): number {
  const expiry = new Date(dateStr)
  const now = new Date()
  const diff = expiry.getTime() - now.getTime()
  return Math.ceil(diff / (1000 * 60 * 60 * 24))
}

// ── Imperial conversions ─────────────────────────────────────────────────────

export const kgToLbs = (kg: number) => Math.round(kg * 2.20462 * 10) / 10

export const lbsToKg = (lbs: number) => Math.round((lbs / 2.20462) * 100) / 100

export function cmToFtIn(cm: number): { ft: number; inches: number } {
  const totalInches = cm / 2.54
  const ft = Math.floor(totalInches / 12)
  const inches = Math.round(totalInches % 12)
  return { ft, inches }
}

export function cmToFtInStr(cm: number): string {
  const { ft, inches } = cmToFtIn(cm)
  return `${ft}'${inches}"`
}

export function ftInToCm(ft: number, inches: number): number {
  return Math.round((ft * 12 + inches) * 2.54 * 10) / 10
}

export function calcAge(dob: string): number | null {
  const birth = new Date(dob)
  if (isNaN(birth.getTime())) return null
  const today = new Date()
  let age = today.getFullYear() - birth.getFullYear()
  const m = today.getMonth() - birth.getMonth()
  if (m < 0 || (m === 0 && today.getDate() < birth.getDate())) age--
  return age
}
