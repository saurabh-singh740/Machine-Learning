import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import Navbar from '@/components/Navbar'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'FL Diabetes — Federated Learning Healthcare',
  description: 'Privacy-preserving diabetes prediction using Federated Learning',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${inter.className} min-h-screen bg-slate-50`}>
        <Navbar />
        <main className="pt-16">{children}</main>
      </body>
    </html>
  )
}
