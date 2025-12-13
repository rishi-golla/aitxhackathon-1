import Header from './components/Header'
import AccidentsCardCompact from './components/AccidentsCardCompact'
import AILearningCard from './components/AILearningCard'
import CamerasPanel from './components/CamerasPanel'
import RecentActivityCard from './components/RecentActivityCard'

export default function DashboardPage() {
  return (
    <main className="flex flex-col flex-1 p-4 gap-3 overflow-hidden h-full">
      {/* Header */}
      <div className="shrink-0">
        <Header />
      </div>
      
      {/* Analytics Row - Top Card (Smaller) */}
      <section className="h-[100px] shrink-0">
        <AccidentsCardCompact />
      </section>
      
      {/* Bottom Section - Cameras (Bigger) and Right Sidebar (Smaller) */}
      <section className="grid grid-cols-[1fr_280px] gap-3 flex-1 min-h-0 overflow-hidden">
        <CamerasPanel />
        <div className="flex flex-col gap-3 min-h-0 h-full overflow-hidden">
          <div className="shrink-0">
            <AILearningCard />
          </div>
          <div className="flex-1 min-h-0 overflow-hidden">
            <RecentActivityCard />
          </div>
        </div>
      </section>
    </main>
  )
}
