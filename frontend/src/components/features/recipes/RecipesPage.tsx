import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import { useAuthStore } from '@/lib/authStore'
import { api } from '@/lib/api'
import { formatDate } from '@/lib/utils'
import type { SavedRecipe } from '@/types'
import PageHeader from '@/components/layout/PageHeader'
import Card from '@/components/ui/Card'
import Button from '@/components/ui/Button'
import Badge from '@/components/ui/Badge'
import Input from '@/components/ui/Input'
import Modal from '@/components/ui/Modal'
import EmptyState from '@/components/ui/EmptyState'
import { useToast } from '@/components/ui/ToastProvider'
import styles from './RecipesPage.module.css'

export default function RecipesPage() {
  const token = useAuthStore((s) => s.accessToken)
  const qc = useQueryClient()
  const { toast } = useToast()
  const [modalOpen, setModalOpen] = useState(false)
  const [selectedRecipe, setSelectedRecipe] = useState<SavedRecipe | null>(null)
  const [form, setForm] = useState({
    title: '',
    instructions: '',
    carbs_per_serving: '',
    calories_per_serving: '',
    servings: '',
    tags: '',
  })

  const { data: recipes, isLoading } = useQuery<SavedRecipe[]>({
    queryKey: ['recipes'],
    queryFn: () => api.get('/recipes', token ?? undefined),
    enabled: !!token,
  })

  const saveMutation = useMutation({
    mutationFn: (body: object) => api.post('/recipes/save', body, token ?? undefined),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['recipes'] })
      setModalOpen(false)
      setForm({ title: '', instructions: '', carbs_per_serving: '', calories_per_serving: '', servings: '', tags: '' })
      toast('Recipe saved', 'success')
    },
    onError: () => toast('Failed to save recipe', 'error'),
  })

  const deleteMutation = useMutation({
    mutationFn: (id: string) => api.delete(`/recipes/${id}`, token ?? undefined),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['recipes'] })
      setSelectedRecipe(null)
      toast('Recipe deleted', 'info')
    },
    onError: () => toast('Failed to delete recipe', 'error'),
  })

  const handleSave = () => {
    if (!form.title) return
    saveMutation.mutate({
      title: form.title,
      instructions: form.instructions,
      ingredients: [],
      carbs_per_serving: form.carbs_per_serving ? parseFloat(form.carbs_per_serving) : undefined,
      calories_per_serving: form.calories_per_serving ? parseFloat(form.calories_per_serving) : undefined,
      servings: form.servings ? parseInt(form.servings) : undefined,
      tags: form.tags ? form.tags.split(',').map((t) => t.trim()).filter(Boolean) : [],
    })
  }

  const set = (f: string) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) =>
    setForm((p) => ({ ...p, [f]: e.target.value }))

  return (
    <div>
      <PageHeader
        title="Recipes"
        subtitle="Your saved, diabetes-friendly recipe collection"
        action={<Button onClick={() => setModalOpen(true)}>Save recipe</Button>}
      />

      {isLoading ? (
        <div className={styles.loading}>Loading your recipes...</div>
      ) : recipes && recipes.length > 0 ? (
        <div className={styles.recipeGrid}>
          <AnimatePresence>
            {recipes.map((recipe, i) => (
              <motion.div
                key={recipe.id}
                className={styles.recipeCard}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.96 }}
                transition={{ delay: i * 0.04 }}
                onClick={() => setSelectedRecipe(recipe)}
              >
                <div className={styles.recipeCardAccent} />
                <div className={styles.recipeCardBody}>
                  <h3 className={styles.recipeTitle}>{recipe.title}</h3>
                  <p className={styles.recipeDate}>Saved {formatDate(recipe.created_at)}</p>

                  {(recipe.carbs_per_serving != null || recipe.calories_per_serving != null) && (
                    <div className={styles.recipeNutrition}>
                      {recipe.carbs_per_serving != null && (
                        <div className={styles.recipeNutritionItem}>
                          <span className={styles.recipeNutritionValue}>{recipe.carbs_per_serving}g</span>
                          <span className={styles.recipeNutritionLabel}>carbs/serving</span>
                        </div>
                      )}
                      {recipe.calories_per_serving != null && (
                        <div className={styles.recipeNutritionItem}>
                          <span className={styles.recipeNutritionValue}>{recipe.calories_per_serving}</span>
                          <span className={styles.recipeNutritionLabel}>kcal/serving</span>
                        </div>
                      )}
                      {recipe.servings != null && (
                        <div className={styles.recipeNutritionItem}>
                          <span className={styles.recipeNutritionValue}>{recipe.servings}</span>
                          <span className={styles.recipeNutritionLabel}>servings</span>
                        </div>
                      )}
                    </div>
                  )}

                  {recipe.tags && recipe.tags.length > 0 && (
                    <div className={styles.recipeTags}>
                      {recipe.tags.slice(0, 3).map((tag) => (
                        <Badge key={tag} variant="default">{tag}</Badge>
                      ))}
                      {recipe.tags.length > 3 && (
                        <span className={styles.moreTagsLabel}>+{recipe.tags.length - 3} more</span>
                      )}
                    </div>
                  )}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      ) : (
        <Card elevated padded>
          <EmptyState
            title="No recipes saved"
            description="Ask TasteMatch to suggest recipes and save them here for quick access."
            icon={<BookIcon />}
          />
        </Card>
      )}

      <Modal open={modalOpen} onClose={() => setModalOpen(false)} title="Save a recipe">
        <div className={styles.modalForm}>
          <Input label="Recipe title" placeholder="e.g. Low-GI salmon stir fry" value={form.title} onChange={set('title')} />
          <div className={styles.row}>
            <Input label="Carbs per serving (g)" type="number" min="0" placeholder="e.g. 30" value={form.carbs_per_serving} onChange={set('carbs_per_serving')} />
            <Input label="Calories per serving" type="number" min="0" placeholder="e.g. 450" value={form.calories_per_serving} onChange={set('calories_per_serving')} />
          </div>
          <div className={styles.row}>
            <Input label="Servings" type="number" min="1" placeholder="e.g. 4" value={form.servings} onChange={set('servings')} />
            <Input label="Tags (comma separated)" placeholder="low-carb, quick, low-gi" value={form.tags} onChange={set('tags')} />
          </div>
          <div className={styles.textareaWrapper}>
            <label className={styles.textareaLabel}>Instructions</label>
            <textarea className={styles.textarea} placeholder="Write your recipe instructions here..." value={form.instructions} onChange={set('instructions')} rows={5} />
          </div>
          {saveMutation.isError && <p className={styles.error}>Failed to save recipe.</p>}
          <div className={styles.modalActions}>
            <Button variant="secondary" onClick={() => setModalOpen(false)}>Cancel</Button>
            <Button onClick={handleSave} loading={saveMutation.isPending}>Save recipe</Button>
          </div>
        </div>
      </Modal>

      <Modal open={!!selectedRecipe} onClose={() => setSelectedRecipe(null)} title={selectedRecipe?.title ?? ''} size="lg">
        {selectedRecipe && (
          <div className={styles.recipeDetail}>
            {(selectedRecipe.carbs_per_serving != null || selectedRecipe.calories_per_serving != null) && (
              <div className={styles.recipeDetailNutrition}>
                {selectedRecipe.carbs_per_serving != null && (
                  <div className={styles.recipeDetailStat}>
                    <span className={styles.recipeDetailStatValue}>{selectedRecipe.carbs_per_serving}g</span>
                    <span className={styles.recipeDetailStatLabel}>Carbs per serving</span>
                  </div>
                )}
                {selectedRecipe.calories_per_serving != null && (
                  <div className={styles.recipeDetailStat}>
                    <span className={styles.recipeDetailStatValue}>{selectedRecipe.calories_per_serving}</span>
                    <span className={styles.recipeDetailStatLabel}>Calories per serving</span>
                  </div>
                )}
                {selectedRecipe.servings != null && (
                  <div className={styles.recipeDetailStat}>
                    <span className={styles.recipeDetailStatValue}>{selectedRecipe.servings}</span>
                    <span className={styles.recipeDetailStatLabel}>Servings</span>
                  </div>
                )}
              </div>
            )}
            {selectedRecipe.tags?.length > 0 && (
              <div className={styles.recipeTags}>
                {selectedRecipe.tags.map((tag) => (
                  <Badge key={tag} variant="default">{tag}</Badge>
                ))}
              </div>
            )}
            {selectedRecipe.instructions && (
              <div className={styles.recipeInstructions}>
                <h4 className={styles.recipeInstructionsTitle}>Instructions</h4>
                <p className={styles.recipeInstructionsText}>{selectedRecipe.instructions}</p>
              </div>
            )}
            <div className={styles.recipeDetailActions}>
              <Button variant="danger" size="sm" onClick={() => deleteMutation.mutate(selectedRecipe.id)} loading={deleteMutation.isPending}>
                Delete recipe
              </Button>
            </div>
          </div>
        )}
      </Modal>
    </div>
  )
}

function BookIcon() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H19a1 1 0 0 1 1 1v18a1 1 0 0 1-1 1H6.5a1 1 0 0 1 0-5H20"/>
    </svg>
  )
}
