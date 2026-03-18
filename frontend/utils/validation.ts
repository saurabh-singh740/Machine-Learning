export interface FieldBounds {
  min: number
  max: number
  label: string
}

export const FIELD_BOUNDS: Record<string, FieldBounds> = {
  pregnancies:      { min: 0,   max: 20,  label: 'Pregnancies' },
  glucose:          { min: 0,   max: 300, label: 'Glucose' },
  blood_pressure:   { min: 0,   max: 200, label: 'Blood Pressure' },
  skin_thickness:   { min: 0,   max: 100, label: 'Skin Thickness' },
  insulin:          { min: 0,   max: 900, label: 'Insulin' },
  bmi:              { min: 0,   max: 70,  label: 'BMI' },
  diabetes_pedigree:{ min: 0,   max: 3,   label: 'Diabetes Pedigree' },
  age:              { min: 1,   max: 120, label: 'Age' },
}

export function validateField(key: string, value: string): string | null {
  if (value.trim() === '') return 'This field is required'
  const n = parseFloat(value)
  if (isNaN(n)) return 'Must be a valid number'
  const bounds = FIELD_BOUNDS[key]
  if (bounds && (n < bounds.min || n > bounds.max)) {
    return `${bounds.label} must be between ${bounds.min} and ${bounds.max}`
  }
  return null
}
