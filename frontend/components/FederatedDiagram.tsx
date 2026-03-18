'use client'
import { useState } from 'react'

const STEPS = [
  {
    id: 0,
    title: 'Step 1 — Data Partitioning',
    desc: 'The global dataset is split using Dirichlet(α) distribution across clients. Each client receives a non-IID subset simulating real hospital populations. The server reserves a held-out test set.',
    highlight: 'clients',
  },
  {
    id: 1,
    title: 'Step 2 — Broadcast Global Model',
    desc: 'The server broadcasts the current global model weights to all participating clients. In round 1 this is a randomly initialized model. Subsequent rounds use the aggregated model.',
    highlight: 'server-to-clients',
  },
  {
    id: 2,
    title: 'Step 3 — Local Training',
    desc: 'Each client trains on their local data for E epochs using Adam optimizer + Cosine LR. FedProx adds a proximal term to prevent drift. DP-SGD optionally adds calibrated Gaussian noise.',
    highlight: 'clients',
  },
  {
    id: 3,
    title: 'Step 4 — Upload Updates',
    desc: 'Clients compute SHA-512 hash of their weight tensors and send (weights, metrics, hash) to the server. No raw data leaves the client — only model parameter deltas.',
    highlight: 'clients-to-server',
  },
  {
    id: 4,
    title: 'Step 5 — Aggregation',
    desc: 'Server verifies SHA-512 hashes, then computes weighted FedAvg: w_global = Σ (n_i / N) · w_i. FedProx clients include proximal term in their local objective (μ/2)·‖w−w_g‖².',
    highlight: 'server',
  },
  {
    id: 5,
    title: 'Step 6 — Server Evaluation',
    desc: 'The aggregated global model is evaluated on the server held-out test set. AUC-ROC, F1, Recall, Precision are computed. The best model checkpoint is saved. Metrics are logged to TensorBoard.',
    highlight: 'server',
  },
]

const clients = ['Hospital A\n(Client 0)', 'Hospital B\n(Client 1)', 'Hospital C\n(Client 2)']

function ClientBox({ label, active }: { label: string; active: boolean }) {
  return (
    <div className={`rounded-xl border-2 p-4 text-center transition-all duration-500 ${
      active ? 'bg-blue-100 border-blue-400 shadow-lg scale-105' : 'bg-slate-100 border-slate-300'
    }`}>
      <div className="text-2xl mb-1">🏥</div>
      <div className="text-xs font-semibold text-slate-700 whitespace-pre-line">{label}</div>
      {active && <div className="mt-2 w-2 h-2 rounded-full bg-blue-500 animate-pulse mx-auto" />}
    </div>
  )
}

export default function FederatedDiagram() {
  const [step, setStep] = useState(0)
  const current = STEPS[step]

  const clientsActive = current.highlight === 'clients' || current.highlight === 'clients-to-server'
  const serverActive = current.highlight === 'server' || current.highlight === 'server-to-clients'
  const arrowDown = current.highlight === 'server-to-clients'
  const arrowUp = current.highlight === 'clients-to-server'

  return (
    <div>
      {/* Step navigator */}
      <div className="flex gap-2 flex-wrap mb-8">
        {STEPS.map(s => (
          <button key={s.id} onClick={() => setStep(s.id)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              step === s.id ? 'bg-teal-600 text-white' : 'bg-white border border-slate-300 text-slate-600 hover:bg-slate-50'
            }`}>
            Step {s.id + 1}
          </button>
        ))}
      </div>

      {/* Diagram */}
      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-8 mb-6">
        <div className="flex flex-col md:flex-row items-center justify-center gap-8">

          {/* Clients column */}
          <div className="flex flex-col gap-3 w-40">
            {clients.map((c, i) => (
              <ClientBox key={i} label={c} active={clientsActive} />
            ))}
          </div>

          {/* Middle arrows */}
          <div className="flex flex-col items-center gap-3 text-slate-400 w-40">
            {arrowUp && (
              <div className="flex flex-col items-center gap-1 text-green-600">
                <span className="text-xs font-mono bg-green-50 border border-green-200 rounded px-2 py-1">weights + SHA-512</span>
                <span className="text-2xl">↑↑↑</span>
              </div>
            )}
            {arrowDown && (
              <div className="flex flex-col items-center gap-1 text-blue-600">
                <span className="text-2xl">↓↓↓</span>
                <span className="text-xs font-mono bg-blue-50 border border-blue-200 rounded px-2 py-1">global model</span>
              </div>
            )}
            {!arrowUp && !arrowDown && (
              <div className="w-20 border-t-2 border-dashed border-slate-300 hidden md:block" />
            )}
          </div>

          {/* Server */}
          <div className={`rounded-2xl border-2 p-6 text-center w-48 transition-all duration-500 ${
            serverActive ? 'bg-blue-600 border-blue-700 text-white shadow-xl scale-105' : 'bg-slate-100 border-slate-300 text-slate-700'
          }`}>
            <div className="text-3xl mb-2">🖥️</div>
            <div className="font-bold text-lg">FL Server</div>
            <div className={`text-xs mt-2 ${serverActive ? 'text-blue-200' : 'text-slate-500'}`}>
              FedAvg / FedProx<br />Aggregator
            </div>
            <div className={`mt-3 rounded-lg px-3 py-2 text-xs ${serverActive ? 'bg-blue-500/40 text-blue-100' : 'bg-slate-200 text-slate-600'}`}>
              SHA-512 Verify<br />AUC-ROC Eval
            </div>
            {serverActive && <div className="mt-2 w-2 h-2 rounded-full bg-white animate-pulse mx-auto" />}
          </div>
        </div>
      </div>

      {/* Step description */}
      <div className="bg-teal-50 border border-teal-200 rounded-2xl p-6">
        <h3 className="text-lg font-bold text-teal-800 mb-2">{current.title}</h3>
        <p className="text-slate-700">{current.desc}</p>
      </div>

      {/* Navigation */}
      <div className="flex justify-between mt-6">
        <button onClick={() => setStep(s => Math.max(0, s - 1))} disabled={step === 0}
          className="px-6 py-2 bg-white border border-slate-300 text-slate-600 rounded-lg disabled:opacity-40 hover:bg-slate-50 transition-colors">
          ← Previous
        </button>
        <span className="text-sm text-slate-500 self-center">
          {step + 1} / {STEPS.length}
        </span>
        <button onClick={() => setStep(s => Math.min(STEPS.length - 1, s + 1))} disabled={step === STEPS.length - 1}
          className="px-6 py-2 bg-teal-600 text-white rounded-lg disabled:opacity-40 hover:bg-teal-700 transition-colors">
          Next →
        </button>
      </div>

      {/* Architecture facts */}
      <div className="mt-10 grid grid-cols-1 md:grid-cols-3 gap-4">
        {[
          { title: 'Data Never Leaves', desc: 'Raw patient records stay inside each hospital node. Only model parameter tensors are transmitted.' },
          { title: 'SHA-512 Integrity', desc: 'Each parameter update includes a cryptographic hash. Server verifies before aggregation.' },
          { title: 'Optional DP Noise', desc: 'Gradient clipping (C) + Gaussian noise (σ) added at each client for (ε,δ)-differential privacy.' },
        ].map((f, i) => (
          <div key={i} className="bg-white rounded-xl p-5 border border-slate-200">
            <div className="font-bold text-slate-800 mb-2">{f.title}</div>
            <p className="text-sm text-slate-500">{f.desc}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
