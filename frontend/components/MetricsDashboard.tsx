'use client'
import { useEffect, useState } from 'react'
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, PointElement,
  LineElement, Title, Tooltip, Legend, Filler,
} from 'chart.js'
import { Line } from 'react-chartjs-2'
import { fetchMetrics, MetricsData } from '@/services/api'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler)

const MOCK: MetricsData = {
  rounds: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
  auc_roc: [0.71, 0.74, 0.76, 0.78, 0.79, 0.81, 0.82, 0.83, 0.835, 0.84],
  f1: [0.60, 0.63, 0.65, 0.67, 0.68, 0.70, 0.71, 0.72, 0.725, 0.73],
  recall: [0.62, 0.65, 0.67, 0.69, 0.70, 0.72, 0.73, 0.74, 0.745, 0.75],
  accuracy: [0.72, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.79, 0.80, 0.80],
  final: { auc_roc: 0.84, f1: 0.73, recall: 0.75, accuracy: 0.80 },
}

function StatCard({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className={`rounded-xl p-5 ${color}`}>
      <p className="text-sm font-medium opacity-80 mb-1">{label}</p>
      <p className="text-3xl font-bold">{(value * 100).toFixed(1)}%</p>
    </div>
  )
}

const chartOptions = {
  responsive: true,
  plugins: { legend: { position: 'top' as const }, tooltip: { mode: 'index' as const } },
  scales: {
    y: { min: 0.5, max: 1.0, ticks: { callback: (v: unknown) => `${((v as number) * 100).toFixed(0)}%` } },
    x: { title: { display: true, text: 'FL Round' } },
  },
}

export default function MetricsDashboard() {
  const [metrics, setMetrics] = useState<MetricsData | null>(null)
  const [loading, setLoading] = useState(true)
  const [usingMock, setUsingMock] = useState(false)

  useEffect(() => {
    fetchMetrics()
      .then(data => setMetrics(data))
      .catch(() => { setMetrics(MOCK); setUsingMock(true) })
      .finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div className="flex items-center justify-center py-24 text-slate-500">
      <svg className="w-6 h-6 animate-spin mr-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="10" strokeOpacity="0.25" /><path d="M12 2a10 10 0 0 1 10 10" />
      </svg>
      Loading metrics…
    </div>
  )

  if (!metrics) return null
  const labels = metrics.rounds.map(r => `Round ${r}`)

  const lineData = {
    labels,
    datasets: [
      { label: 'AUC-ROC', data: metrics.auc_roc, borderColor: '#2563eb', backgroundColor: 'rgba(37,99,235,0.08)', tension: 0.4, fill: true },
      { label: 'F1 Score', data: metrics.f1, borderColor: '#7c3aed', backgroundColor: 'rgba(124,58,237,0.08)', tension: 0.4, fill: true },
      { label: 'Recall', data: metrics.recall, borderColor: '#059669', backgroundColor: 'rgba(5,150,105,0.08)', tension: 0.4, fill: true },
      { label: 'Accuracy', data: metrics.accuracy, borderColor: '#d97706', backgroundColor: 'rgba(217,119,6,0.08)', tension: 0.4, fill: true },
    ],
  }

  return (
    <div>
      {usingMock && (
        <div className="mb-6 bg-amber-50 border border-amber-200 rounded-lg px-4 py-3 text-sm text-amber-700">
          <strong>Demo mode:</strong> Showing simulated metrics. Run the FL system and start the API server to see live data.
        </div>
      )}

      {/* Stat cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <StatCard label="AUC-ROC" value={metrics.final.auc_roc} color="bg-blue-600 text-white" />
        <StatCard label="F1 Score" value={metrics.final.f1} color="bg-purple-600 text-white" />
        <StatCard label="Recall" value={metrics.final.recall} color="bg-green-600 text-white" />
        <StatCard label="Accuracy" value={metrics.final.accuracy} color="bg-amber-500 text-white" />
      </div>

      {/* Training curve */}
      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 mb-6">
        <h3 className="text-lg font-bold text-slate-800 mb-4">Training Curves — All Metrics per Round</h3>
        <Line data={lineData} options={chartOptions} />
      </div>

      {/* AUC-ROC only */}
      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
        <h3 className="text-lg font-bold text-slate-800 mb-4">AUC-ROC Convergence</h3>
        <Line
          data={{ labels, datasets: [lineData.datasets[0]] }}
          options={{
            ...chartOptions,
            scales: {
              ...chartOptions.scales,
              y: { min: 0.65, max: 0.90, ticks: { callback: (v: unknown) => `${((v as number) * 100).toFixed(0)}%` } },
            },
          }}
        />
      </div>
    </div>
  )
}
