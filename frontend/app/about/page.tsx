export default function AboutPage() {
  return (
    <div className="min-h-screen bg-slate-50">
      <div className="bg-gradient-to-r from-slate-800 to-slate-700 text-white py-14 px-6">
        <div className="max-w-5xl mx-auto text-center">
          <h1 className="text-4xl font-bold mb-3">About This Project</h1>
          <p className="text-slate-300">
            Technical overview of the system, dataset, and federated learning methodology.
          </p>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-6 py-12 space-y-12">

        {/* Diabetes */}
        <section className="bg-white rounded-2xl p-8 shadow-sm border border-slate-200">
          <h2 className="text-2xl font-bold text-slate-800 mb-4">🩺 Type 2 Diabetes</h2>
          <p className="text-slate-600 mb-4">
            Type 2 Diabetes affects over 400 million people worldwide. Early prediction using clinical biomarkers
            allows for preventive intervention — but training accurate models requires large, diverse patient cohorts
            that no single hospital can provide alone.
          </p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
            {[
              { label: 'Global Prevalence', value: '~537M', sub: 'people with diabetes' },
              { label: 'Undiagnosed', value: '~45%', sub: 'remain undetected' },
              { label: 'Prevention', value: '58%', sub: 'reducible with lifestyle changes' },
              { label: 'Dataset', value: '768', sub: 'Pima Indian samples' },
            ].map((s, i) => (
              <div key={i} className="bg-blue-50 rounded-xl p-4 text-center">
                <div className="text-2xl font-bold text-blue-700">{s.value}</div>
                <div className="text-xs font-semibold text-slate-700 mt-1">{s.label}</div>
                <div className="text-xs text-slate-500">{s.sub}</div>
              </div>
            ))}
          </div>
        </section>

        {/* Dataset */}
        <section className="bg-white rounded-2xl p-8 shadow-sm border border-slate-200">
          <h2 className="text-2xl font-bold text-slate-800 mb-4">📋 Pima Indians Diabetes Dataset</h2>
          <p className="text-slate-600 mb-6">
            768 samples, 8 clinical features, binary outcome (diabetic / non-diabetic). Class distribution: ~65% negative, ~35% positive.
            Class imbalance is addressed using <code className="bg-slate-100 px-1 rounded">BCEWithLogitsLoss(pos_weight)</code>.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-sm border-collapse">
              <thead className="bg-slate-100">
                <tr>
                  <th className="px-4 py-2 text-left text-slate-700">Feature</th>
                  <th className="px-4 py-2 text-left text-slate-700">Description</th>
                  <th className="px-4 py-2 text-left text-slate-700">Range</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ['Pregnancies', 'Number of pregnancies', '0 – 17'],
                  ['Glucose', 'Plasma glucose (mg/dL)', '0 – 199'],
                  ['Blood Pressure', 'Diastolic BP (mm Hg)', '0 – 122'],
                  ['Skin Thickness', 'Triceps skinfold (mm)', '0 – 99'],
                  ['Insulin', '2-hour serum insulin (µU/mL)', '0 – 846'],
                  ['BMI', 'Body mass index (kg/m²)', '0 – 67.1'],
                  ['Pedigree Function', 'Genetic diabetes risk score', '0.078 – 2.42'],
                  ['Age', 'Age in years', '21 – 81'],
                ].map(([f, d, r], i) => (
                  <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-slate-50'}>
                    <td className="px-4 py-2 font-medium text-blue-700">{f}</td>
                    <td className="px-4 py-2 text-slate-600">{d}</td>
                    <td className="px-4 py-2 text-slate-500 font-mono text-xs">{r}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* FL Methodology */}
        <section className="bg-white rounded-2xl p-8 shadow-sm border border-slate-200">
          <h2 className="text-2xl font-bold text-slate-800 mb-4">🔬 Federated Learning Methodology</h2>
          <div className="grid md:grid-cols-2 gap-6">
            {[
              { title: 'FedAvg', content: 'McMahan et al. (2017). Clients train locally; server aggregates weighted average of parameters. Weight proportional to local dataset size.' },
              { title: 'FedProx', content: 'Li et al. (2020). Adds proximal term μ/2 · ‖w − w_global‖² to local objective. Prevents client drift under non-IID data distributions.' },
              { title: 'Dirichlet Non-IID', content: 'Data partitioned via Dir(α). α=0.5 produces realistic hospital heterogeneity. Lower α creates more extreme non-IID scenarios.' },
              { title: 'Differential Privacy', content: 'DP-SGD: gradient clipping to norm C + Gaussian noise N(0, σC/B). Privacy budget (ε, δ) tracked via moments accountant.' },
            ].map((item, i) => (
              <div key={i} className="bg-slate-50 rounded-xl p-5">
                <h3 className="font-bold text-slate-800 mb-2">{item.title}</h3>
                <p className="text-sm text-slate-600">{item.content}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Model Architecture */}
        <section className="bg-white rounded-2xl p-8 shadow-sm border border-slate-200">
          <h2 className="text-2xl font-bold text-slate-800 mb-6">🧠 Model Architecture</h2>
          <div className="flex flex-col items-center gap-2">
            {[
              { label: 'Input', detail: '8 clinical features', color: 'bg-blue-100 border-blue-300 text-blue-800' },
              { label: 'Linear(8→64) + BatchNorm + ReLU + Dropout(0.3)', detail: '', color: 'bg-slate-100 border-slate-300 text-slate-700' },
              { label: 'Linear(64→32) + BatchNorm + ReLU + Dropout(0.3)', detail: '', color: 'bg-slate-100 border-slate-300 text-slate-700' },
              { label: 'Linear(32→16) + BatchNorm + ReLU + Dropout(0.15)', detail: '', color: 'bg-slate-100 border-slate-300 text-slate-700' },
              { label: 'Linear(16→1)', detail: 'raw logit', color: 'bg-slate-100 border-slate-300 text-slate-700' },
              { label: 'Sigmoid → Probability [0,1]', detail: 'inference only', color: 'bg-green-100 border-green-300 text-green-800' },
            ].map((layer, i) => (
              <div key={i} className="w-full max-w-lg">
                <div className={`border-2 rounded-lg px-4 py-3 text-center text-sm font-mono font-medium ${layer.color}`}>
                  {layer.label}
                  {layer.detail && <span className="text-xs opacity-60 ml-2">← {layer.detail}</span>}
                </div>
                {i < 5 && <div className="text-center text-slate-400 text-xl">↓</div>}
              </div>
            ))}
          </div>
          <p className="text-center text-sm text-slate-500 mt-6">Total trainable parameters: ~4,273 | Optimizer: Adam + Cosine LR</p>
        </section>

        {/* References */}
        <section className="bg-white rounded-2xl p-8 shadow-sm border border-slate-200">
          <h2 className="text-2xl font-bold text-slate-800 mb-4">📚 References</h2>
          <ol className="space-y-3 text-sm text-slate-600">
            {[
              'McMahan, H. B., et al. (2017). Communication-efficient learning of deep networks from decentralized data. AISTATS 2017.',
              'Li, T., et al. (2020). Federated optimization in heterogeneous networks. MLSys 2020.',
              'Abadi, M., et al. (2016). Deep learning with differential privacy. CCS 2016.',
              'Beutel, D. J., et al. (2020). Flower: A friendly federated learning research framework. arXiv:2007.14390.',
              'Smith, V., et al. (2017). Federated multi-task learning. NeurIPS 2017.',
            ].map((ref, i) => (
              <li key={i} className="flex gap-3">
                <span className="font-bold text-blue-600 shrink-0">[{i + 1}]</span>
                <span>{ref}</span>
              </li>
            ))}
          </ol>
        </section>
      </div>
    </div>
  )
}
