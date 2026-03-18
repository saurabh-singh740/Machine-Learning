import { PredictionResult } from '@/services/api'

interface Props {
  result: PredictionResult
  className?: string
}

const riskConfig = {
  Low: { color: 'text-green-700', bg: 'bg-green-50', border: 'border-green-300', bar: 'bg-green-500', emoji: '✅' },
  Moderate: { color: 'text-yellow-700', bg: 'bg-yellow-50', border: 'border-yellow-300', bar: 'bg-yellow-500', emoji: '⚠️' },
  Elevated: { color: 'text-orange-700', bg: 'bg-orange-50', border: 'border-orange-300', bar: 'bg-orange-500', emoji: '🔶' },
  High: { color: 'text-red-700', bg: 'bg-red-50', border: 'border-red-300', bar: 'bg-red-500', emoji: '🚨' },
} as const

type RiskLevel = keyof typeof riskConfig

export default function ResultCard({ result, className = '' }: Props) {
  const level = (result.risk_level as RiskLevel) || 'Low'
  const cfg = riskConfig[level] ?? riskConfig.Low
  const pct = Math.round(result.probability * 100)

  return (
    <div className={`bg-white rounded-2xl shadow-sm border-2 ${cfg.border} p-8 ${className}`}>
      <h3 className="text-lg font-bold text-slate-800 mb-6">Prediction Result</h3>

      {/* Probability bar */}
      <div className="mb-6">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-slate-600">Diabetes Probability</span>
          <span className="text-2xl font-bold text-slate-800">{pct}%</span>
        </div>
        <div className="w-full bg-slate-100 rounded-full h-4 overflow-hidden">
          <div
            className={`h-4 rounded-full transition-all duration-700 ${cfg.bar}`}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {/* Risk level badge */}
      <div className={`${cfg.bg} ${cfg.border} border rounded-xl px-5 py-4 flex items-center gap-4 mb-6`}>
        <span className="text-3xl">{cfg.emoji}</span>
        <div>
          <div className={`text-xl font-bold ${cfg.color}`}>{level} Risk</div>
          <div className="text-sm text-slate-500">Based on federated model prediction</div>
        </div>
      </div>

      {/* Recommendation */}
      {result.recommendation && (
        <div className="bg-slate-50 rounded-xl px-5 py-4 border border-slate-200">
          <p className="text-sm font-semibold text-slate-700 mb-1">Clinical Recommendation</p>
          <p className="text-sm text-slate-600">{result.recommendation}</p>
        </div>
      )}

      {/* Metadata */}
      <div className="mt-5 grid grid-cols-2 gap-3 text-xs text-slate-500">
        {result.threshold !== undefined && (
          <div className="bg-slate-50 rounded-lg px-3 py-2">
            <span className="font-medium text-slate-700">Decision Threshold:</span> {result.threshold.toFixed(3)}
          </div>
        )}
        {result.model_version && (
          <div className="bg-slate-50 rounded-lg px-3 py-2">
            <span className="font-medium text-slate-700">Model:</span> {result.model_version}
          </div>
        )}
      </div>

      <p className="text-xs text-slate-400 mt-4 italic">
        ⚕ This prediction is for research purposes only and does not constitute medical advice.
        Please consult a qualified healthcare professional.
      </p>
    </div>
  )
}
