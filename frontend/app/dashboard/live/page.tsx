'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'

interface Camera {
  id: string
  label: string
  streamUrl: string
  floor: number
  status: 'normal' | 'warning' | 'incident' | 'inactive'
  active: boolean
}

const GridIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
    <rect x="2" y="2" width="5" height="5" rx="1" />
    <rect x="9" y="2" width="5" height="5" rx="1" />
    <rect x="2" y="9" width="5" height="5" rx="1" />
    <rect x="9" y="9" width="5" height="5" rx="1" />
  </svg>
)

const ListIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M2 4H14M2 8H14M2 12H14" strokeLinecap="round" />
  </svg>
)

const FullscreenIcon = () => (
  <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M2 5V2H5M11 2H14V5M14 11V14H11M5 14H2V11" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
)

const StatusBadge = ({ status }: { status: string }) => {
  const colors = {
    normal: 'bg-teal-500/20 text-teal-400 border-teal-500/30',
    warning: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
    incident: 'bg-red-500/20 text-red-400 border-red-500/30',
    inactive: 'bg-gray-500/20 text-gray-400 border-gray-500/30',
  }
  
  return (
    <div className={`px-2 py-0.5 rounded text-[10px] font-medium border ${colors[status as keyof typeof colors]}`}>
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </div>
  )
}

const CameraFeedCard = ({ camera, layout }: { camera: Camera; layout: 'grid' | 'list' }) => {
  const [isHovered, setIsHovered] = useState(false)
  
  if (layout === 'list') {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="rounded-xl bg-card border border-white/5 p-4 hover:border-cyan/20 transition-all duration-300 flex items-center gap-4"
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        {/* Video Preview */}
        <div className="w-64 h-36 rounded-lg bg-zinc-900 relative overflow-hidden shrink-0">
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <div className="w-12 h-12 mx-auto mb-2 rounded-full bg-white/5 flex items-center justify-center">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polygon points="5 3 19 12 5 21 5 3" fill="currentColor" opacity="0.3" />
                </svg>
              </div>
              <p className="text-xs text-white/40">Stream: {camera.streamUrl}</p>
            </div>
          </div>
          
          {/* Live Badge */}
          <div className="absolute top-2 left-2 px-2 py-1 rounded bg-red-500 text-white text-[10px] font-bold flex items-center gap-1">
            <div className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
            LIVE
          </div>
          
          {/* Fullscreen Button */}
          {isHovered && (
            <motion.button
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="absolute top-2 right-2 p-2 rounded-lg bg-black/50 hover:bg-black/70 text-white transition-colors backdrop-blur-sm"
            >
              <FullscreenIcon />
            </motion.button>
          )}
        </div>
        
        {/* Info */}
        <div className="flex-1">
          <div className="flex items-start justify-between mb-2">
            <div>
              <h3 className="text-white font-semibold text-base mb-1">{camera.label}</h3>
              <p className="text-white/50 text-xs">Floor {camera.floor}</p>
            </div>
            <StatusBadge status={camera.status} />
          </div>
          
          <div className="flex items-center gap-4 text-xs text-white/40 mt-3">
            <div className="flex items-center gap-1">
              <div className="w-1.5 h-1.5 rounded-full bg-green-400" />
              Connected
            </div>
            <div>1920x1080</div>
            <div>30 FPS</div>
          </div>
        </div>
      </motion.div>
    )
  }
  
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className="rounded-xl bg-card border border-white/5 p-3 hover:border-cyan/20 transition-all duration-300"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Video Feed */}
      <div className="aspect-video rounded-lg bg-zinc-900 relative overflow-hidden mb-3">
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <div className="w-10 h-10 mx-auto mb-2 rounded-full bg-white/5 flex items-center justify-center">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polygon points="5 3 19 12 5 21 5 3" fill="currentColor" opacity="0.3" />
              </svg>
            </div>
            <p className="text-[10px] text-white/30">Stream: {camera.streamUrl}</p>
          </div>
        </div>
        
        {/* Live Badge */}
        <div className="absolute top-2 left-2 px-2 py-0.5 rounded bg-red-500 text-white text-[9px] font-bold flex items-center gap-1">
          <div className="w-1 h-1 rounded-full bg-white animate-pulse" />
          LIVE
        </div>
        
        {/* Fullscreen Button */}
        {isHovered && (
          <motion.button
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="absolute top-2 right-2 p-1.5 rounded-lg bg-black/50 hover:bg-black/70 text-white transition-colors backdrop-blur-sm"
          >
            <FullscreenIcon />
          </motion.button>
        )}
      </div>
      
      {/* Info */}
      <div className="flex items-start justify-between mb-2">
        <div>
          <h3 className="text-white font-medium text-sm mb-0.5">{camera.label}</h3>
          <p className="text-white/50 text-[10px]">Floor {camera.floor}</p>
        </div>
        <StatusBadge status={camera.status} />
      </div>
      
      <div className="flex items-center gap-2 text-[9px] text-white/40">
        <div className="flex items-center gap-1">
          <div className="w-1 h-1 rounded-full bg-green-400" />
          Connected
        </div>
      </div>
    </motion.div>
  )
}

export default function LivePage() {
  const [layout, setLayout] = useState<'grid' | 'list'>('grid')
  const [selectedFloor, setSelectedFloor] = useState<number | 'all'>('all')
  const [cameras, setCameras] = useState<Camera[]>([])
  
  // Load cameras from localStorage
  useEffect(() => {
    const stored = localStorage.getItem('factory-floorplan-vectors')
    if (stored) {
      try {
        const data = JSON.parse(stored)
        const loadedCameras: Camera[] = data.cameras.map((cam: any) => ({
          ...cam,
          status: ['normal', 'warning', 'incident', 'inactive'][Math.floor(Math.random() * 4)],
        }))
        setCameras(loadedCameras)
      } catch (error) {
        console.error('Failed to load cameras:', error)
      }
    }
  }, [])
  
  const filteredCameras = selectedFloor === 'all' 
    ? cameras 
    : cameras.filter(cam => cam.floor === selectedFloor)
  
  const gridCols = layout === 'grid' ? 'grid-cols-3' : 'grid-cols-1'
  
  return (
    <div className="flex-1 p-6 overflow-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-semibold text-white mb-1">Live Cameras</h1>
          <p className="text-white/50 text-sm">
            {filteredCameras.length} camera{filteredCameras.length !== 1 ? 's' : ''} active
          </p>
        </div>
        
        <div className="flex items-center gap-3">
          {/* Floor Filter */}
          <div className="flex items-center gap-2 px-4 py-2 bg-card border border-white/10 rounded-lg">
            <span className="text-sm text-zinc-400">Floor:</span>
            <button
              onClick={() => setSelectedFloor('all')}
              className={`px-3 py-1 rounded-lg text-sm font-medium transition-all ${
                selectedFloor === 'all'
                  ? 'bg-cyan text-dark'
                  : 'text-zinc-400 hover:text-white hover:bg-white/5'
              }`}
            >
              All
            </button>
            {[1, 2, 3, 4, 5].map((floor) => (
              <button
                key={floor}
                onClick={() => setSelectedFloor(floor)}
                className={`px-3 py-1 rounded-lg text-sm font-medium transition-all ${
                  selectedFloor === floor
                    ? 'bg-cyan text-dark'
                    : 'text-zinc-400 hover:text-white hover:bg-white/5'
                }`}
              >
                {floor}
              </button>
            ))}
          </div>
          
          {/* Layout Toggle */}
          <div className="flex items-center gap-1 p-1 bg-card border border-white/10 rounded-lg">
            <button
              onClick={() => setLayout('grid')}
              className={`p-2 rounded transition-all ${
                layout === 'grid'
                  ? 'bg-cyan text-dark'
                  : 'text-zinc-400 hover:text-white'
              }`}
            >
              <GridIcon />
            </button>
            <button
              onClick={() => setLayout('list')}
              className={`p-2 rounded transition-all ${
                layout === 'list'
                  ? 'bg-cyan text-dark'
                  : 'text-zinc-400 hover:text-white'
              }`}
            >
              <ListIcon />
            </button>
          </div>
        </div>
      </div>
      
      {/* Camera Grid/List */}
      {filteredCameras.length === 0 ? (
        <div className="flex items-center justify-center h-96">
          <div className="text-center">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-white/5 flex items-center justify-center">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
                <circle cx="12" cy="13" r="4" />
              </svg>
            </div>
            <h3 className="text-white text-lg font-medium mb-2">No Cameras Found</h3>
            <p className="text-white/50 text-sm mb-4">
              {selectedFloor === 'all' 
                ? 'Add cameras in the Camera Layout page to see them here.'
                : `No cameras on floor ${selectedFloor}.`}
            </p>
          </div>
        </div>
      ) : (
        <div className={`grid ${gridCols} gap-4`}>
          {filteredCameras.map((camera, index) => (
            <CameraFeedCard key={camera.id} camera={camera} layout={layout} />
          ))}
        </div>
      )}
    </div>
  )
}

