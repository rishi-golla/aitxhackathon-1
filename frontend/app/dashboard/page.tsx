import Sidebar from './components/Sidebar'
import Header from './components/Header'
import AccidentsCardCompact from './components/AccidentsCardCompact'
import AILearningCard from './components/AILearningCard'
import CamerasPanel from './components/CamerasPanel'
import RecentActivityCard from './components/RecentActivityCard'

export default function DashboardPage() {
  return (
    <div className="flex h-screen bg-dark overflow-hidden">
      {/* Sidebar */}
      <Sidebar />
      
      {/* Main Content */}
      <main className="flex flex-col flex-1 p-4 bg-gradient-to-br from-dark to-dark-secondary gap-3 overflow-hidden">
        {/* Header */}
        <div className="shrink-0">
          <Header />
        </div>
        
        {/* Analytics Row - Top Card */}
        <section className="h-[130px] shrink-0">
          <AccidentsCardCompact />
        </section>
        
        {/* Bottom Section - Cameras and Right Sidebar */}
        <section className="grid grid-cols-[1fr_320px] gap-3 flex-1 min-h-0">
          <CamerasPanel />
          <div className="flex flex-col gap-3 min-h-0">
            <div className="shrink-0">
              <AILearningCard />
            </div>
            <div className="flex-1 min-h-0">
              <RecentActivityCard />
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}
