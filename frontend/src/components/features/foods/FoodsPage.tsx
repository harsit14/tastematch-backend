import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import { useToken } from '@/hooks/useToken'
import { useDebounce } from '@/hooks/useDebounce'
import { api } from '@/lib/api'
import type { Food } from '@/types'
import PageHeader from '@/components/layout/PageHeader'
import Card from '@/components/ui/Card'
import Badge from '@/components/ui/Badge'
import { Skeleton } from '@/components/ui/Skeleton'
import EmptyState from '@/components/ui/EmptyState'
import styles from './FoodsPage.module.css'

interface FoodsResponse {
  source: 'local' | 'api'
  results: Food[]
}

const GI_VARIANT: Record<string, 'success' | 'warning' | 'danger'> = {
  low: 'success',
  medium: 'warning',
  high: 'danger',
}

export default function FoodsPage() {
  const token = useToken()
  const [query, setQuery] = useState('')
  const debouncedQuery = useDebounce(query, 450)

  const { data, isLoading, isFetching } = useQuery<FoodsResponse>({
    queryKey: ['foods-search', debouncedQuery],
    queryFn: () => api.get(`/foods/search?q=${encodeURIComponent(debouncedQuery)}&limit=12`, token),
    enabled: debouncedQuery.length >= 2,
    staleTime: 1000 * 60 * 5,
  })

  const showSkeleton = (isLoading || isFetching) && debouncedQuery.length >= 2
  const showEmpty = !showSkeleton && debouncedQuery.length >= 2 && (!data?.results || data.results.length === 0)
  const showResults = !showSkeleton && data?.results && data.results.length > 0

  return (
    <div>
      <PageHeader
        title="Food Search"
        subtitle="Search nutritional data from USDA and Open Food Facts"
      />

      <div className={styles.searchBar}>
        <div className={styles.searchInputWrapper}>
          <span className={styles.searchIcon}><SearchIcon /></span>
          <input
            className={styles.searchInput}
            type="search"
            placeholder="Search for a food, ingredient, or product..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            autoFocus
          />
          {isFetching && <span className={styles.searchSpinner} />}
        </div>
        {data?.source && showResults && (
          <span className={styles.sourceTag}>
            Source: {data.source === 'local' ? 'local cache' : 'USDA / Open Food Facts'}
          </span>
        )}
      </div>

      {query.length < 2 && (
        <div className={styles.hint}>
          <div className={styles.hintGrid}>
            {EXAMPLE_SEARCHES.map((s) => (
              <button key={s} className={styles.hintChip} onClick={() => setQuery(s)}>
                {s}
              </button>
            ))}
          </div>
        </div>
      )}

      {showSkeleton && (
        <div className={styles.resultsGrid}>
          {Array.from({ length: 6 }).map((_, i) => (
            <FoodCardSkeleton key={i} />
          ))}
        </div>
      )}

      {showEmpty && (
        <EmptyState
          title="No results found"
          description={`No foods matched "${debouncedQuery}". Try a different search term.`}
          icon={<SearchIcon />}
        />
      )}

      {showResults && (
        <div className={styles.resultsGrid}>
          <AnimatePresence>
            {data!.results.map((food, i) => (
              <FoodCard key={food.id ?? food.external_id ?? i} food={food} index={i} />
            ))}
          </AnimatePresence>
        </div>
      )}
    </div>
  )
}

function FoodCard({ food, index }: { food: Food; index: number }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <motion.div
      className={styles.foodCard}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.96 }}
      transition={{ delay: index * 0.035 }}
      onClick={() => setExpanded((p) => !p)}
    >
      <div className={styles.foodCardHeader}>
        <div className={styles.foodCardMeta}>
          <h3 className={styles.foodName}>{food.name}</h3>
          {food.brand && <span className={styles.foodBrand}>{food.brand}</span>}
        </div>
        <div className={styles.foodCardBadges}>
          {food.gi_category && (
            <Badge variant={GI_VARIANT[food.gi_category]}>GI: {food.gi_category}</Badge>
          )}
          {food.source && (
            <Badge variant="muted">{food.source === 'usda' ? 'USDA' : food.source === 'open_food_facts' ? 'OFF' : food.source}</Badge>
          )}
        </div>
      </div>

      <div className={styles.macroRow}>
        <MacroCell label="Energy" value={food.calories_per_100g} unit="kcal" />
        <MacroCell label="Carbs" value={food.carbs_per_100g} unit="g" highlight />
        <MacroCell label="Protein" value={food.protein_per_100g} unit="g" />
        <MacroCell label="Fat" value={food.fat_per_100g} unit="g" />
      </div>

      <AnimatePresence>
        {expanded && (
          <motion.div
            className={styles.expandedDetails}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
          >
            <div className={styles.expandedGrid}>
              <DetailItem label="Fibre" value={food.fiber_per_100g != null ? `${food.fiber_per_100g}g` : '--'} />
              <DetailItem label="Serving size" value={food.serving_size_g != null ? `${food.serving_size_g}g` : '--'} />
              <DetailItem label="Glycaemic index" value={food.glycemic_index != null ? String(food.glycemic_index) : '--'} />
              <DetailItem label="GI category" value={food.gi_category ?? '--'} />
            </div>
            <p className={styles.expandedNote}>Values per 100g unless otherwise stated.</p>
          </motion.div>
        )}
      </AnimatePresence>

      <button className={styles.expandToggle} aria-label={expanded ? 'Show less' : 'Show more'}>
        <ChevronIcon rotated={expanded} />
      </button>
    </motion.div>
  )
}

function MacroCell({ label, value, unit, highlight = false }: {
  label: string
  value?: number | null
  unit: string
  highlight?: boolean
}) {
  return (
    <div className={styles.macroCell}>
      <span className={`${styles.macroValue} ${highlight ? styles.macroHighlight : ''}`}>
        {value != null ? value.toFixed(1) : '--'}
      </span>
      <span className={styles.macroUnit}>{unit}</span>
      <span className={styles.macroLabel}>{label}</span>
    </div>
  )
}

function DetailItem({ label, value }: { label: string; value: string }) {
  return (
    <div className={styles.detailItem}>
      <span className={styles.detailLabel}>{label}</span>
      <span className={styles.detailValue}>{value}</span>
    </div>
  )
}

function FoodCardSkeleton() {
  return (
    <div className={styles.foodCard}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 16 }}>
        <Skeleton height={16} width="70%" />
        <Skeleton height={12} width="40%" />
      </div>
      <div style={{ display: 'flex', gap: 12 }}>
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} height={40} width={56} />
        ))}
      </div>
    </div>
  )
}

function ChevronIcon({ rotated }: { rotated: boolean }) {
  return (
    <motion.svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      animate={{ rotate: rotated ? 180 : 0 }}
      transition={{ duration: 0.2 }}
    >
      <polyline points="6 9 12 15 18 9" />
    </motion.svg>
  )
}

function SearchIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  )
}

const EXAMPLE_SEARCHES = [
  'chicken breast', 'brown rice', 'sweet potato', 'salmon',
  'oats', 'lentils', 'avocado', 'Greek yogurt',
]
