import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import { useAuthStore } from '@/lib/authStore'
import { api } from '@/lib/api'
import { daysUntilExpiry, formatDate } from '@/lib/utils'
import type { FridgeItem } from '@/types'
import PageHeader from '@/components/layout/PageHeader'
import Card from '@/components/ui/Card'
import Button from '@/components/ui/Button'
import Badge from '@/components/ui/Badge'
import Input from '@/components/ui/Input'
import Modal from '@/components/ui/Modal'
import EmptyState from '@/components/ui/EmptyState'
import { useToast } from '@/components/ui/ToastProvider'
import styles from './FridgePage.module.css'

export default function FridgePage() {
  const token = useAuthStore((s) => s.accessToken)
  const qc = useQueryClient()
  const { toast } = useToast()
  const [modalOpen, setModalOpen] = useState(false)
  const [form, setForm] = useState({
    food_name: '',
    quantity: '',
    unit: '',
    expiry_date: '',
  })

  const { data: items, isLoading } = useQuery<FridgeItem[]>({
    queryKey: ['fridge'],
    queryFn: () => api.get('/fridge', token ?? undefined),
    enabled: !!token,
  })

  const addMutation = useMutation({
    mutationFn: (body: object) => api.post('/fridge', body, token ?? undefined),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['fridge'] })
      setModalOpen(false)
      setForm({ food_name: '', quantity: '', unit: '', expiry_date: '' })
      toast('Item added to fridge', 'success')
    },
    onError: () => toast('Failed to add item', 'error'),
  })

  const removeMutation = useMutation({
    mutationFn: (id: string) => api.delete(`/fridge/${id}`, token ?? undefined),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['fridge'] })
      toast('Item removed', 'info')
    },
    onError: () => toast('Failed to remove item', 'error'),
  })

  const handleAdd = () => {
    if (!form.food_name) return
    addMutation.mutate({
      food_name: form.food_name,
      quantity: form.quantity ? parseFloat(form.quantity) : undefined,
      unit: form.unit || undefined,
      expiry_date: form.expiry_date || undefined,
    })
  }

  const set = (f: string) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm((p) => ({ ...p, [f]: e.target.value }))

  const expiryVariant = (days: number): 'danger' | 'warning' | 'success' => {
    if (days <= 2) return 'danger'
    if (days <= 5) return 'warning'
    return 'success'
  }

  const soonExpiring = items?.filter((i) => i.expiry_date && daysUntilExpiry(i.expiry_date) <= 3) ?? []

  return (
    <div>
      <PageHeader
        title="Fridge"
        subtitle="Keep track of what you have on hand for smarter meal suggestions"
        action={<Button onClick={() => setModalOpen(true)}>Add item</Button>}
      />

      {soonExpiring.length > 0 && (
        <div className={styles.expiryAlert}>
          <div className={styles.expiryAlertIcon}><ClockIcon /></div>
          <div className={styles.expiryAlertText}>
            <strong>Expiring soon</strong>
            <p>
              {soonExpiring.map((i) => i.food_name).join(', ')} will expire within 3 days.
              Ask TasteMatch for recipe ideas to use them up.
            </p>
          </div>
        </div>
      )}

      <Card elevated>
        <Card.Header
          title="Your fridge contents"
          subtitle={items ? `${items.length} item${items.length !== 1 ? 's' : ''}` : undefined}
        />

        {isLoading ? (
          <div className={styles.loading}>Loading your fridge...</div>
        ) : items && items.length > 0 ? (
          <ul className={styles.itemGrid}>
            <AnimatePresence>
              {items.map((item, i) => {
                const days = item.expiry_date ? daysUntilExpiry(item.expiry_date) : null
                return (
                  <motion.li
                    key={item.id}
                    className={styles.fridgeItem}
                    initial={{ opacity: 0, scale: 0.97 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95, x: -20 }}
                    transition={{ delay: i * 0.03 }}
                    layout
                  >
                    <div className={styles.itemIcon}>
                      <FoodIcon />
                    </div>
                    <div className={styles.itemBody}>
                      <span className={styles.itemName}>{item.food_name}</span>
                      <div className={styles.itemMeta}>
                        {item.quantity != null && (
                          <span className={styles.itemQty}>
                            {item.quantity}{item.unit ? ` ${item.unit}` : ''}
                          </span>
                        )}
                        {item.expiry_date && days !== null && (
                          <Badge variant={expiryVariant(days)}>
                            {days <= 0 ? 'Expired' : days === 1 ? 'Expires tomorrow' : `${days}d left`}
                          </Badge>
                        )}
                        {item.expiry_date && (
                          <span className={styles.itemDate}>Exp. {formatDate(item.expiry_date)}</span>
                        )}
                      </div>
                    </div>
                    <button
                      className={styles.removeBtn}
                      onClick={() => removeMutation.mutate(item.id)}
                      aria-label={`Remove ${item.food_name}`}
                    >
                      <TrashIcon />
                    </button>
                  </motion.li>
                )
              })}
            </AnimatePresence>
          </ul>
        ) : (
          <EmptyState
            title="Your fridge is empty"
            description="Add items to get personalised recipe suggestions based on what you have on hand."
            icon={<ArchiveIcon />}
          />
        )}
      </Card>

      <Modal open={modalOpen} onClose={() => setModalOpen(false)} title="Add fridge item">
        <div className={styles.modalForm}>
          <Input
            label="Item name"
            placeholder="e.g. Chicken breast"
            value={form.food_name}
            onChange={set('food_name')}
          />
          <div className={styles.row}>
            <Input
              label="Quantity (optional)"
              type="number"
              min="0"
              placeholder="e.g. 300"
              value={form.quantity}
              onChange={set('quantity')}
            />
            <Input
              label="Unit (optional)"
              placeholder="e.g. g, ml, pieces"
              value={form.unit}
              onChange={set('unit')}
            />
          </div>
          <Input
            label="Expiry date (optional)"
            type="date"
            value={form.expiry_date}
            onChange={set('expiry_date')}
          />
          {addMutation.isError && (
            <p className={styles.error}>Failed to add item. Please try again.</p>
          )}
          <div className={styles.modalActions}>
            <Button variant="secondary" onClick={() => setModalOpen(false)}>Cancel</Button>
            <Button onClick={handleAdd} loading={addMutation.isPending}>Add to fridge</Button>
          </div>
        </div>
      </Modal>
    </div>
  )
}

function FoodIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2z"/><path d="M12 8v4l3 3"/>
    </svg>
  )
}
function ClockIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
    </svg>
  )
}
function TrashIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
    </svg>
  )
}
function ArchiveIcon() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/>
      <path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/>
    </svg>
  )
}
