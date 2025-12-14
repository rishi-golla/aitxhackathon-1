'use client'

import { useEffect } from 'react'
import Sidebar from './components/Sidebar'

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  // Auto-login with hardcoded demo user
  useEffect(() => {
    // Set a demo token in localStorage so the app thinks we're logged in
    const demoToken = 'demo-user-token-12345'
    const demoUser = {
      id: 'demo-user-id',
      email: 'demo@example.com',
      fullName: 'Demo User',
      company: 'Demo Company',
      role: 'admin',
      avatar: 'https://api.dicebear.com/7.x/initials/svg?seed=Demo User',
    }
    
    localStorage.setItem('token', demoToken)
    localStorage.setItem('user', JSON.stringify(demoUser))
  }, [])

  return (
    <div className="flex h-screen bg-dark overflow-hidden">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        {children}
      </div>
    </div>
  )
}
