'use client'
import PredictionForm from '@/components/PredictionForm'

export default function PredictionPage() {
  return (
    <div className="min-h-screen bg-slate-50">
      <div className="bg-gradient-to-r from-blue-800 to-blue-700 text-white py-14 px-6">
        <div className="max-w-3xl mx-auto text-center">
          <h1 className="text-4xl font-bold mb-3">Diabetes Risk Prediction</h1>
          <p className="text-blue-200">
            Enter patient clinical data below. The federated model will predict the likelihood of Type 2 Diabetes.
          </p>
        </div>
      </div>
      <div className="max-w-3xl mx-auto px-6 py-12">
        <PredictionForm />
      </div>
    </div>
  )
}
