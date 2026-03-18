'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'

const links = [
  { href: '/', label: 'Home' },
  { href: '/prediction', label: 'Prediction' },
  { href: '/metrics', label: 'Metrics' },
  { href: '/visualization', label: 'FL Diagram' },
  { href: '/about', label: 'About' },
]

export default function Navbar() {
  const pathname = usePathname()
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-blue-900/95 backdrop-blur-sm border-b border-blue-800 shadow-lg">
      <div className="max-w-6xl mx-auto px-6 flex items-center justify-between h-16">
        <Link href="/" className="flex items-center gap-2 text-white font-bold text-lg">
          <span className="text-2xl">🧬</span>
          <span>FL Diabetes</span>
        </Link>
        <div className="flex gap-1">
          {links.map(({ href, label }) => (
            <Link key={href} href={href}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                pathname === href
                  ? 'bg-blue-600 text-white'
                  : 'text-blue-200 hover:text-white hover:bg-blue-800'
              }`}>
              {label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  )
}
