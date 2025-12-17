import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Pose Fitness Coach',
  description: 'AI-powered fitness coaching using pose detection',
  icons: {
    icon: '/favicon.ico',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  )
}
