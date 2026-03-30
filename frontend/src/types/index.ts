export interface User {
  id: string
  email: string
}

export interface UserProfile {
  id: string
  first_name: string
  last_name: string
  age: number
  date_of_birth: string
  diabetes_type: 'type1' | 'type2' | 'prediabetes' | 'gestational' | 'other' | 'none'
  dietary_preferences: string[]
  allergies: string[]
  created_at: string
  updated_at: string
}

export interface GlucoseReading {
  id: string
  user_id: string
  reading_mmol: number
  reading_context: 'fasting' | 'before_meal' | 'after_meal' | 'bedtime' | 'other'
  notes?: string
  recorded_at: string
  created_at: string
}

export interface MealLog {
  id: string
  user_id: string
  meal_name: string
  carbs_grams: number
  calories?: number
  meal_type?: 'breakfast' | 'lunch' | 'dinner' | 'snack'
  estimated_gi?: number
  glycemic_load?: number
  notes?: string
  eaten_at: string
  created_at: string
}

export interface BodyMetric {
  id: string
  user_id: string
  weight_kg: number
  height_cm: number
  bmi: number
  recorded_at: string
}

export interface FridgeItem {
  id: string
  user_id: string
  food_id?: string
  food_name: string
  quantity?: number
  unit?: string
  expiry_date?: string
  added_at: string
}

export interface Food {
  id: string
  external_id?: string
  source?: string
  name: string
  brand?: string
  calories_per_100g?: number
  protein_per_100g?: number
  carbs_per_100g?: number
  fat_per_100g?: number
  fiber_per_100g?: number
  serving_size_g?: number
  glycemic_index?: number
  gi_category?: 'low' | 'medium' | 'high'
}

export interface SavedRecipe {
  id: string
  user_id: string
  title: string
  ingredients: RecipeIngredient[]
  instructions: string
  carbs_per_serving?: number
  calories_per_serving?: number
  servings?: number
  tags: string[]
  created_at: string
}

export interface RecipeIngredient {
  name: string
  quantity: string
  unit: string
}

export interface ChatSession {
  id: string
  user_id: string
  title: string
  created_at: string
  updated_at: string
}

export interface ChatRecipe {
  title: string
  instructions: string
  ingredients: string[]
  tags: string[]
  carbs_per_serving: number | null
  calories_per_serving: number | null
  servings: number | null
}

export interface ChatMessage {
  id: string
  session_id: string
  role: 'user' | 'assistant'
  content: string
  created_at: string
  recipe?: ChatRecipe
}

export interface AuthState {
  user: User | null
  accessToken: string | null
  isAuthenticated: boolean
}
