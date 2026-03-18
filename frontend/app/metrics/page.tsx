'use client'
import MetricsDashboard from '@/components/MetricsDashboard'

export default function MetricsPage() {
  return (
    <div className="min-h-screen bg-slate-50">
      <div className="bg-gradient-to-r from-purple-800 to-purple-700 text-white py-14 px-6">
        <div className="max-w-5xl mx-auto text-center">
          <h1 className="text-4xl font-bold mb-3">Metrics Dashboard</h1>
          <p className="text-purple-200">
            Real-time training metrics from the federated learning experiment.
          </p>
        </div>
      </div>
      <div className="max-w-6xl mx-auto px-6 py-12">
        <MetricsDashboard />
      </div>
    </div>
  )
}
