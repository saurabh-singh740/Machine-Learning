'use client'
import { useState } from 'react'
import { predictDiabetes, PredictionInput, PredictionResult } from '@/services/api'
import ResultCard from './ResultCard'

const fields = [
  { key: 'pregnancies', label: 'Pregnancies', min: 0, max: 20, step: 1, unit: 'count', placeholder: '0 – 17' },
  { key: 'glucose', label: 'Glucose', min: 0, max: 300, step: 1, unit: 'mg/dL', placeholder: '70 – 199' },
  { key: 'blood_pressure', label: 'Blood Pressure', min: 0, max: 200, step: 1, unit: 'mm Hg', placeholder: '40 – 122' },
  { key: 'skin_thickness', label: 'Skin Thickness', min: 0, max: 100, step: 1, unit: 'mm', placeholder: '10 – 99' },
  { key: 'insulin', label: 'Insulin', min: 0, max: 900, step: 1, unit: 'µU/mL', placeholder: '0 – 846' },
  { key: 'bmi', label: 'BMI', min: 0, max: 70, step: 0.1, unit: 'kg/m²', placeholder: '18.5 – 67.1' },
  { key: 'diabetes_pedigree', label: 'Diabetes Pedigree Function', min: 0, max: 3, step: 0.001, unit: 'score', placeholder: '0.078 – 2.42' },
  { key: 'age', label: 'Age', min: 1, max: 120, step: 1, unit: 'years', placeholder: '21 – 81' },
] as const

type FieldKey = typeof fields[number]['key']
type FormValues = Record<FieldKey, string>

const emptyForm: FormValues = {
  pregnancies: '', glucose: '', blood_pressure: '', skin_thickness: '',
  insulin: '', bmi: '', diabetes_pedigree: '', age: '',
}

const examplePatient: FormValues = {
  pregnancies: '2', glucose: '148', blood_pressure: '72', skin_thickness: '35',
  insulin: '0', bmi: '33.6', diabetes_pedigree: '0.627', age: '50',
}

function validate(values: FormValues): Partial<Record<FieldKey, string>> {
  const errors: Partial<Record<FieldKey, string>> = {}
  for (const f of fields) {
    const raw = values[f.key].trim()
    if (raw === '') { errors[f.key] = 'Required'; continue }
    const n = parseFloat(raw)
    if (isNaN(n)) { errors[f.key] = 'Must be a number'; continue }
    if (n < f.min || n > f.max) { errors[f.key] = `Must be between ${f.min} and ${f.max}`; continue }
  }
  return errors
}

export default function PredictionForm() {
  const [values, setValues] = useState<FormValues>(emptyForm)
  const [errors, setErrors] = useState<Partial<Record<FieldKey, string>>>({})
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [apiError, setApiError] = useState<string | null>(null)

  const handleChange = (key: FieldKey, val: string) => {
    setValues(prev => ({ ...prev, [key]: val }))
    if (errors[key]) setErrors(prev => ({ ...prev, [key]: undefined }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setApiError(null)
    const errs = validate(values)
    if (Object.keys(errs).length > 0) { setErrors(errs); return }

    setLoading(true)
    setResult(null)
    try {
      const input: PredictionInput = {
        pregnancies: parseFloat(values.pregnancies),
        glucose: parseFloat(values.glucose),
        blood_pressure: parseFloat(values.blood_pressure),
        skin_thickness: parseFloat(values.skin_thickness),
        insulin: parseFloat(values.insulin),
        bmi: parseFloat(values.bmi),
        diabetes_pedigree: parseFloat(values.diabetes_pedigree),
        age: parseFloat(values.age),
      }
      const res = await predictDiabetes(input)
      setResult(res)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Prediction failed'
      setApiError(msg)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setValues(emptyForm)
    setErrors({})
    setResult(null)
    setApiError(null)
  }

  return (
    <div>
      <form onSubmit={handleSubmit} noValidate className="bg-white rounded-2xl shadow-sm border border-slate-200 p-8">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold text-slate-800">Patient Clinical Data</h2>
          <button type="button" onClick={() => setValues(examplePatient)}
            className="text-sm text-blue-600 hover:text-blue-800 underline">
            Load Example
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {fields.map(f => (
            <div key={f.key}>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                {f.label} <span className="text-slate-400 font-normal">({f.unit})</span>
              </label>
              <input
                type="number"
                value={values[f.key]}
                onChange={e => handleChange(f.key, e.target.value)}
                min={f.min}
                max={f.max}
                step={f.step}
                placeholder={f.placeholder}
                className={`w-full border rounded-lg px-4 py-2.5 text-slate-800 text-sm focus:outline-none focus:ring-2 transition-colors ${
                  errors[f.key]
                    ? 'border-red-400 focus:ring-red-200 bg-red-50'
                    : 'border-slate-300 focus:ring-blue-200 focus:border-blue-400'
                }`}
              />
              {errors[f.key] && (
                <p className="text-xs text-red-600 mt-1">{errors[f.key]}</p>
              )}
            </div>
          ))}
        </div>

        {apiError && (
          <div className="mt-5 bg-red-50 border border-red-200 rounded-lg px-4 py-3 text-sm text-red-700">
            <strong>Error:</strong> {apiError}
            {apiError.includes('Network') && (
              <p className="mt-1 text-xs text-red-500">Make sure the Flask API is running at <code>http://localhost:8000</code></p>
            )}
          </div>
        )}

        <div className="flex gap-3 mt-6">
          <button type="submit" disabled={loading}
            className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white font-semibold py-3 rounded-xl transition-colors flex items-center justify-center gap-2">
            {loading ? (
              <>
                <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" strokeOpacity="0.25" />
                  <path d="M12 2a10 10 0 0 1 10 10" />
                </svg>
                Predicting…
              </>
            ) : 'Run Prediction'}
          </button>
          <button type="button" onClick={handleReset}
            className="px-6 py-3 border border-slate-300 text-slate-600 hover:bg-slate-50 rounded-xl transition-colors text-sm font-medium">
            Reset
          </button>
        </div>
      </form>

      {result && <ResultCard result={result} className="mt-8" />}
    </div>
  )
}
