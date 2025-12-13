'use client'

import dynamic from 'next/dynamic'

const DigitalTwinManual = dynamic(() => import('./DigitalTwinManual'), { 
  ssr: false,
  loading: () => (
    <div className="w-full h-full rounded-xl bg-zinc-950 flex items-center justify-center">
      <div className="text-zinc-400 text-sm">Loading 3D environment...</div>
    </div>
  )
})

const ArrowIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M6 4L10 8L6 12" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
)

export default function CamerasPanel() {
  return (
    <div className="rounded-2xl bg-card shadow-card border border-white/5 p-4 hover:shadow-card-hover hover:border-cyan/20 transition-all duration-300 flex flex-col animate-slide-up min-h-0" style={{ animationDelay: '300ms' }}>
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <h3 className="text-white text-base font-semibold">Camera Layout</h3>
        <button className="text-white/30 hover:text-cyan transition-colors">
          <ArrowIcon />
        </button>
      </div>
      
      {/* 3D Digital Twin Container */}
      <div className="relative flex-1 min-h-0 rounded-xl overflow-hidden border border-white/10">
        <DigitalTwinManual />
      </div>
    </div>
  )
}
