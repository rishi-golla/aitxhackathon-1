import Sidebar from './components/Sidebar'

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="flex h-screen bg-dark overflow-hidden">
      <Sidebar />
      <div className="flex-1 bg-gradient-to-br from-dark to-dark-secondary overflow-hidden">
        {children}
      </div>
    </div>
  )
}

