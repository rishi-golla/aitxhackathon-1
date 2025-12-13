'use client'

import { useEffect, useState } from 'react'

const ArrowIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M6 4L10 8L6 12" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
)

interface ActivityEvent {
  id: number
  type: 'accident' | 'connection' | 'lost'
  title: string
  subtitle: string
  dotColor: 'pink' | 'white'
}

const AccidentIcon = () => (
  <div className="w-8 h-8 rounded-lg bg-white/5 flex items-center justify-center shrink-0 border border-white/10">
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
      <path d="M8 2L13 11H3L8 2Z" stroke="#EF4444" strokeWidth="1.2" strokeLinejoin="round"/>
      <path d="M8 5V7.5" stroke="#EF4444" strokeWidth="1.2" strokeLinecap="round"/>
      <circle cx="8" cy="9.5" r="0.5" fill="#EF4444"/>
    </svg>
  </div>
)

const ConnectionIcon = () => (
  <div className="w-8 h-8 rounded-lg bg-white/5 flex items-center justify-center shrink-0 border border-white/10">
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
      <circle cx="8" cy="8" r="4" stroke="#60FFD1" strokeWidth="1.2"/>
      <path d="M8 4V8L10 9.5" stroke="#60FFD1" strokeWidth="1.2" strokeLinecap="round"/>
    </svg>
  </div>
)

const LostIcon = () => (
  <div className="w-8 h-8 rounded-lg bg-white/5 flex items-center justify-center shrink-0 border border-white/10">
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
      <path d="M4 4L12 12M12 4L4 12" stroke="#6B7280" strokeWidth="1.2" strokeLinecap="round"/>
    </svg>
  </div>
)

const EventIcon = ({ type }: { type: string }) => {
  switch (type) {
    case 'accident': return <AccidentIcon />
    case 'connection': return <ConnectionIcon />
    default: return <LostIcon />
  }
}

export default function RecentActivityCard() {
  const [visibleItems, setVisibleItems] = useState<number[]>([])
  
  const activities: ActivityEvent[] = [
    { id: 1, type: 'accident', title: 'New accident detected:', subtitle: 'Crowd on camera N°2', dotColor: 'pink' },
    { id: 2, type: 'connection', title: 'Connection renewed:', subtitle: 'camera N°25', dotColor: 'pink' },
    { id: 3, type: 'accident', title: 'New accident detected:', subtitle: 'Smoke on camera N°17', dotColor: 'pink' },
    { id: 4, type: 'lost', title: 'Lost connection:', subtitle: 'camera N°25', dotColor: 'pink' },
    { id: 5, type: 'lost', title: 'Lost connection:', subtitle: 'camera N°17', dotColor: 'pink' },
  ]
  
  useEffect(() => {
    activities.forEach((_, index) => {
      setTimeout(() => {
        setVisibleItems(prev => [...prev, activities[index].id])
      }, index * 100)
    })
  }, [])
  
  return (
    <div className="w-full rounded-2xl bg-card shadow-card border border-white/5 p-4 hover:shadow-card-hover hover:border-cyan/20 transition-all duration-300 flex flex-col animate-slide-up flex-1 min-h-0" style={{ animationDelay: '400ms' }}>
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <h3 className="text-white/80 text-xs font-medium">Recent Activity</h3>
        <button className="text-white/30 hover:text-cyan transition-colors">
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M6 4L10 8L6 12" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
      </div>
      
      {/* Activity List */}
      <div className="flex-1 space-y-2 overflow-y-auto min-h-0 pr-1">
        {activities.map((activity, index) => (
          <div 
            key={activity.id} 
            className={`flex items-center gap-2 p-1.5 rounded-lg hover:bg-white/5 transition-all duration-200 ${
              visibleItems.includes(activity.id) ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-4'
            }`}
            style={{ transitionDelay: `${index * 100}ms` }}
          >
            <div className="w-7 h-7 rounded-lg bg-white/5 flex items-center justify-center shrink-0 border border-white/10">
              {activity.type === 'accident' && (
                <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
                  <path d="M8 2L13 11H3L8 2Z" stroke="#EF4444" strokeWidth="1.2" strokeLinejoin="round"/>
                  <path d="M8 5V7.5" stroke="#EF4444" strokeWidth="1.2" strokeLinecap="round"/>
                  <circle cx="8" cy="9.5" r="0.5" fill="#EF4444"/>
                </svg>
              )}
              {activity.type === 'connection' && (
                <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
                  <circle cx="8" cy="8" r="4" stroke="#60FFD1" strokeWidth="1.2"/>
                  <path d="M8 4V8L10 9.5" stroke="#60FFD1" strokeWidth="1.2" strokeLinecap="round"/>
                </svg>
              )}
              {activity.type === 'lost' && (
                <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
                  <path d="M4 4L12 12M12 4L4 12" stroke="#6B7280" strokeWidth="1.2" strokeLinecap="round"/>
                </svg>
              )}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-[9px] text-white/50 leading-tight">{activity.title}</p>
              <p className="text-[11px] text-white/90 font-medium truncate">{activity.subtitle}</p>
            </div>
            <div 
              className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                activity.dotColor === 'pink' 
                  ? 'bg-[#EA5CA4] shadow-lg shadow-[#EA5CA4]/50' 
                  : 'bg-white/40'
              }`}
            />
          </div>
        ))}
      </div>
      
      {/* Mark as Read Button */}
      <button className="mt-3 w-full py-2 rounded-lg bg-gradient-to-r from-cyan/10 to-accent/10 border border-cyan/30 text-cyan text-[10px] font-medium hover:from-cyan/20 hover:to-accent/20 hover:border-cyan/50 hover:shadow-glow transition-all duration-200">
        Mark as read
      </button>
    </div>
  )
}
