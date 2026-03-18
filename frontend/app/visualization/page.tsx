import FederatedDiagram from '@/components/FederatedDiagram'

export default function VisualizationPage() {
  return (
    <div className="min-h-screen bg-slate-50">
      <div className="bg-gradient-to-r from-teal-800 to-teal-700 text-white py-14 px-6">
        <div className="max-w-5xl mx-auto text-center">
          <h1 className="text-4xl font-bold mb-3">FL System Visualization</h1>
          <p className="text-teal-200">
            Understand how the federated training loop operates across clients and the server.
          </p>
        </div>
      </div>
      <div className="max-w-6xl mx-auto px-6 py-12">
        <FederatedDiagram />
      </div>
    </div>
  )
}
