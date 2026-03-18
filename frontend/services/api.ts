import axios, { AxiosError } from 'axios'

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

const client = axios.create({
  baseURL: API_BASE,
  timeout: 15_000,
  headers: { 'Content-Type': 'application/json' },
})

// ── Types ────────────────────────────────────────────────────────────────────

export interface PredictionInput {
  pregnancies: number
  glucose: number
  blood_pressure: number
  skin_thickness: number
  insulin: number
  bmi: number
  diabetes_pedigree: number
  age: number
}

export interface PredictionResult {
  probability: number
  prediction: 0 | 1
  risk_level: 'Low' | 'Moderate' | 'Elevated' | 'High'
  recommendation: string
  threshold?: number
  model_version?: string
}

export interface MetricsData {
  rounds: number[]
  auc_roc: number[]
  f1: number[]
  recall: number[]
  accuracy: number[]
  final: { auc_roc: number; f1: number; recall: number; accuracy: number }
}

// ── API calls ─────────────────────────────────────────────────────────────────

export async function predictDiabetes(input: PredictionInput): Promise<PredictionResult> {
  try {
    const { data } = await client.post<PredictionResult>('/predict', input)
    return data
  } catch (err) {
    const e = err as AxiosError
    if (e.code === 'ERR_NETWORK' || e.code === 'ECONNREFUSED') {
      throw new Error('Network error — make sure the API server is running at ' + API_BASE)
    }
    if (e.response?.status === 422) {
      throw new Error('Invalid input data. Please check all fields.')
    }
    if (e.response?.status && e.response.status >= 500) {
      throw new Error('Server error. Please try again later.')
    }
    throw new Error(e.message || 'Prediction failed')
  }
}

export async function fetchMetrics(): Promise<MetricsData> {
  const { data } = await client.get<MetricsData>('/metrics')
  return data
}

export async function checkHealth(): Promise<boolean> {
  try {
    await client.get('/health', { timeout: 3000 })
    return true
  } catch {
    return false
  }
}
