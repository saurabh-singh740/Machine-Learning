import Link from 'next/link'

export default function HomePage() {
  return (
    <div className="min-h-screen">
      {/* Hero */}
      <section className="bg-gradient-to-br from-blue-900 via-blue-800 to-blue-700 text-white py-24 px-6">
        <div className="max-w-5xl mx-auto text-center">
          <div className="inline-flex items-center gap-2 bg-blue-600/40 border border-blue-400/30 rounded-full px-4 py-1.5 text-sm mb-6">
            <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></span>
            Privacy-Preserving Machine Learning
          </div>
          <h1 className="text-5xl font-bold mb-6 leading-tight">
            Federated Learning for<br />
            <span className="text-blue-300">Type 2 Diabetes</span> Prediction
          </h1>
          <p className="text-xl text-blue-100 max-w-3xl mx-auto mb-10">
            A distributed ML system where hospitals collaboratively train a shared model
            without ever sharing patient data — ensuring privacy, compliance, and accuracy.
          </p>
          <div className="flex flex-wrap gap-4 justify-center">
            <Link href="/prediction"
              className="bg-white text-blue-900 font-semibold px-8 py-3 rounded-lg hover:bg-blue-50 transition-colors">
              Run Prediction
            </Link>
            <Link href="/metrics"
              className="border border-white/40 text-white font-semibold px-8 py-3 rounded-lg hover:bg-white/10 transition-colors">
              View Metrics
            </Link>
          </div>
        </div>
      </section>

      {/* How it works */}
      <section className="py-20 px-6 bg-white">
        <div className="max-w-5xl mx-auto">
          <h2 className="text-3xl font-bold text-center text-slate-800 mb-4">How Federated Learning Works</h2>
          <p className="text-slate-500 text-center mb-14 max-w-2xl mx-auto">
            Each hospital keeps patient data local. Only encrypted model updates travel to the server.
          </p>

          {/* Diagram */}
          <div className="flex flex-col md:flex-row items-center justify-center gap-6">
            {/* Hospitals */}
            <div className="flex flex-col gap-4">
              {['Hospital A', 'Hospital B', 'Hospital C'].map((h, i) => (
                <div key={i} className="bg-blue-50 border border-blue-200 rounded-xl px-6 py-4 text-center w-44">
                  <div className="text-2xl mb-1">🏥</div>
                  <div className="font-semibold text-blue-800 text-sm">{h}</div>
                  <div className="text-xs text-slate-500 mt-1">Local Patient Data</div>
                </div>
              ))}
            </div>

            {/* Arrows */}
            <div className="flex flex-col items-center gap-2 text-slate-400 md:flex-col">
              <div className="text-xs font-mono bg-slate-100 rounded px-3 py-1 border">model weights →</div>
              <div className="w-16 border-t-2 border-dashed border-blue-300 hidden md:block"></div>
              <div className="text-xs font-mono bg-slate-100 rounded px-3 py-1 border">← global model</div>
            </div>

            {/* Server */}
            <div className="bg-gradient-to-b from-blue-600 to-blue-700 text-white rounded-2xl px-8 py-8 text-center w-52 shadow-xl">
              <div className="text-3xl mb-2">🖥️</div>
              <div className="font-bold text-lg">FL Server</div>
              <div className="text-xs text-blue-200 mt-2">FedAvg / FedProx<br />Aggregation</div>
              <div className="mt-3 bg-blue-500/40 rounded-lg px-3 py-2 text-xs">
                SHA-512 Integrity<br />Server-side Eval
              </div>
            </div>

            {/* Right arrow */}
            <div className="text-2xl text-blue-400 hidden md:block">→</div>

            {/* Global Model */}
            <div className="bg-green-50 border-2 border-green-300 rounded-2xl px-6 py-8 text-center w-44">
              <div className="text-3xl mb-2">🧠</div>
              <div className="font-bold text-green-800 text-sm">Global Model</div>
              <div className="text-xs text-slate-500 mt-2">Aggregated<br />Knowledge</div>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="py-20 px-6 bg-slate-50">
        <div className="max-w-5xl mx-auto">
          <h2 className="text-3xl font-bold text-center text-slate-800 mb-12">System Features</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              { icon: '🔒', title: 'Differential Privacy', desc: 'DP-SGD with gradient clipping and Gaussian noise. Configurable ε-δ privacy budget.' },
              { icon: '⚖️', title: 'FedAvg & FedProx', desc: 'Two aggregation strategies. FedProx adds proximal regularization for non-IID data.' },
              { icon: '📊', title: 'Clinical Metrics', desc: 'AUC-ROC, Recall, F1, AUC-PR. Youden\'s J threshold selection. Full evaluation reports.' },
              { icon: '🔐', title: 'SHA-512 Integrity', desc: 'Every model update is hashed. Server verifies integrity before aggregation.' },
              { icon: '🏥', title: 'Non-IID Data', desc: 'Dirichlet(α) partitioning simulates real-world heterogeneous hospital populations.' },
              { icon: '📈', title: 'Experiment Tracking', desc: 'TensorBoard + JSON logging. Training curves, convergence plots, per-client metrics.' },
            ].map((f, i) => (
              <div key={i} className="bg-white rounded-xl p-6 shadow-sm border border-slate-200">
                <div className="text-3xl mb-3">{f.icon}</div>
                <h3 className="font-bold text-slate-800 mb-2">{f.title}</h3>
                <p className="text-sm text-slate-500">{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Results table */}
      <section className="py-20 px-6 bg-white">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold text-center text-slate-800 mb-4">Expected Performance</h2>
          <p className="text-slate-500 text-center mb-10">Pima Indians Diabetes Dataset — 768 samples, 8 features</p>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr className="bg-blue-600 text-white">
                  <th className="px-6 py-3 text-left rounded-tl-lg">System</th>
                  <th className="px-6 py-3 text-center">AUC-ROC</th>
                  <th className="px-6 py-3 text-center">F1 Score</th>
                  <th className="px-6 py-3 text-center rounded-tr-lg">Recall</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ['Centralised Baseline', '0.83 – 0.87', '0.70 – 0.76', '0.72 – 0.80', 'bg-green-50'],
                  ['FL FedAvg (IID)', '0.80 – 0.85', '0.67 – 0.74', '0.70 – 0.78', 'bg-white'],
                  ['FL FedProx (non-IID)', '0.79 – 0.84', '0.66 – 0.73', '0.69 – 0.77', 'bg-white'],
                  ['FL + DP (σ=1.0)', '0.76 – 0.82', '0.63 – 0.70', '0.66 – 0.74', 'bg-slate-50'],
                ].map(([sys, auc, f1, rec, bg], i) => (
                  <tr key={i} className={`${bg} border-b border-slate-200`}>
                    <td className="px-6 py-4 font-medium text-slate-700">{sys}</td>
                    <td className="px-6 py-4 text-center text-blue-700 font-mono">{auc}</td>
                    <td className="px-6 py-4 text-center text-slate-600 font-mono">{f1}</td>
                    <td className="px-6 py-4 text-center text-slate-600 font-mono">{rec}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-16 px-6 bg-blue-900 text-white text-center">
        <h2 className="text-3xl font-bold mb-4">Ready to predict diabetes risk?</h2>
        <p className="text-blue-200 mb-8">Enter patient clinical data and get an instant risk assessment.</p>
        <Link href="/prediction"
          className="bg-white text-blue-900 font-bold px-10 py-4 rounded-xl hover:bg-blue-50 transition-colors text-lg">
          Start Prediction →
        </Link>
      </section>
    </div>
  )
}
