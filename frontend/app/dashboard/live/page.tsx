'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { HARDCODED_CAMERAS } from '@/lib/hardcoded-data'

interface Camera {
  id: string
  label: string
  streamUrl: string
  floor: number
  status: 'normal' | 'warning' | 'incident' | 'inactive'
  active: boolean
}

interface Violation {
  camera_id: string
  camera_label: string
  code: string
  title: string
  text: string
  penalty: string
}

const BACKEND_URL = 'http://localhost:8000'

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


const DGXBadge = () => (
  <div className="flex items-center gap-1 px-1.5 py-0.5 bg-gradient-to-r from-green-500/20 to-cyan/20 border border-green-500/30 rounded text-[8px] font-bold text-green-400">
    <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke="currentColor" strokeWidth="2" fill="none"/>
    </svg>
    DGX SPARK
  </div>
)

const CameraFeedCard = ({ camera, layout, hasViolation }: { camera: Camera; layout: 'grid' | 'list'; hasViolation?: boolean }) => {
  const [isHovered, setIsHovered] = useState(false)
  const [streamError, setStreamError] = useState(false)
  const [videoSummary, setVideoSummary] = useState<string>('AI analyzing video feed...')
  const [summaryLoading, setSummaryLoading] = useState(true)

  // Build stream URL from backend
  const streamUrl = `${BACKEND_URL}/video_feed/${camera.id}`

  // Fetch real AI summary from backend
  useEffect(() => {
    const fetchSummary = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/summarize/${camera.id}`)
        if (response.ok) {
          const data = await response.json()
          setVideoSummary(data.summary || 'No summary available')
          setSummaryLoading(false)
        }
      } catch (error) {
        console.warn(`Failed to fetch summary for ${camera.id}:`, error)
        setVideoSummary('AI analysis pending... (Connect to DGX backend)')
        setSummaryLoading(false)
      }
    }

    fetchSummary()
    // Refresh summary every 30 seconds
    const interval = setInterval(fetchSummary, 30000)
    return () => clearInterval(interval)
  }, [camera.id])

  if (layout === 'list') {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className={`rounded-xl bg-card border p-4 transition-all duration-300 flex items-center gap-4 ${
          hasViolation ? 'border-red-500/50 shadow-lg shadow-red-500/20' : 'border-white/5 hover:border-cyan/20'
        }`}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        {/* Video Preview - MJPEG Stream */}
        <div className="w-64 h-36 rounded-lg bg-zinc-900 relative overflow-hidden shrink-0">
          {!streamError ? (
            <img
              src={streamUrl}
              alt={camera.label}
              className="w-full h-full object-cover"
              onError={() => setStreamError(true)}
            />
          ) : (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <div className="w-12 h-12 mx-auto mb-2 rounded-full bg-white/5 flex items-center justify-center">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polygon points="5 3 19 12 5 21 5 3" fill="currentColor" opacity="0.3" />
                  </svg>
                </div>
                <p className="text-xs text-white/40">Stream offline</p>
              </div>
            </div>
          )}

          {/* Status Badge */}
          <div className={`absolute top-2 left-2 px-2 py-1 rounded text-white text-[10px] font-bold flex items-center gap-1 ${
            hasViolation ? 'bg-red-600 animate-pulse' : 'bg-zinc-700'
          }`}>
            <div className={`w-1.5 h-1.5 rounded-full ${hasViolation ? 'bg-white animate-pulse' : 'bg-red-500'}`} />
            {hasViolation ? 'VIOLATION' : 'REC'}
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
            <StatusBadge status={hasViolation ? 'incident' : camera.status} />
          </div>

          {/* Video Summary - DGX Superpower */}
          <div className="mb-3 p-2.5 rounded-lg bg-gradient-to-br from-zinc-800/50 to-zinc-900/50 border border-white/5">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[10px] font-semibold text-cyan uppercase tracking-wider">Video Summary</span>
              <DGXBadge />
            </div>
            {summaryLoading ? (
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-cyan animate-pulse" />
                <p className="text-xs text-white/50 italic">Analyzing with DGX Spark...</p>
              </div>
            ) : (
              <p className="text-xs text-white/70 leading-relaxed">{videoSummary}</p>
            )}
          </div>

          <div className="flex items-center gap-4 text-xs text-white/40">
            <div className="flex items-center gap-1">
              <div className={`w-1.5 h-1.5 rounded-full ${streamError ? 'bg-red-400' : 'bg-green-400'}`} />
              {streamError ? 'Disconnected' : 'Connected'}
            </div>
            <div>AI Detection Active</div>
          </div>
        </div>
      </motion.div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`rounded-xl bg-card border p-3 transition-all duration-300 ${
        hasViolation ? 'border-red-500/50 shadow-lg shadow-red-500/20' : 'border-white/5 hover:border-cyan/20'
      }`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Video Feed - MJPEG Stream */}
      <div className="aspect-video rounded-lg bg-zinc-900 relative overflow-hidden mb-3">
        {!streamError ? (
          <img
            src={streamUrl}
            alt={camera.label}
            className="w-full h-full object-cover"
            onError={() => setStreamError(true)}
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center">
              <div className="w-10 h-10 mx-auto mb-2 rounded-full bg-white/5 flex items-center justify-center">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polygon points="5 3 19 12 5 21 5 3" fill="currentColor" opacity="0.3" />
                </svg>
              </div>
              <p className="text-[10px] text-white/30">Stream offline</p>
            </div>
          </div>
        )}

        {/* Status Badge */}
        <div className={`absolute top-2 left-2 px-2 py-0.5 rounded text-white text-[9px] font-bold flex items-center gap-1 ${
          hasViolation ? 'bg-red-600 animate-pulse' : 'bg-zinc-700'
        }`}>
          <div className={`w-1 h-1 rounded-full ${hasViolation ? 'bg-white animate-pulse' : 'bg-red-500'}`} />
          {hasViolation ? 'VIOLATION' : 'REC'}
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
        <StatusBadge status={hasViolation ? 'incident' : camera.status} />
      </div>

      {/* Video Summary - DGX Superpower */}
      <div className="mb-2 p-2 rounded-lg bg-gradient-to-br from-zinc-800/50 to-zinc-900/50 border border-white/5">
        <div className="flex items-center justify-between mb-1">
          <span className="text-[9px] font-semibold text-cyan uppercase tracking-wider">Video Summary</span>
          <DGXBadge />
        </div>
        {summaryLoading ? (
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-cyan animate-pulse" />
            <p className="text-[10px] text-white/50 italic">Analyzing with DGX Spark...</p>
          </div>
        ) : (
          <p className="text-[10px] text-white/70 leading-relaxed">{videoSummary}</p>
        )}
      </div>

      <div className="flex items-center gap-2 text-[9px] text-white/40">
        <div className="flex items-center gap-1">
          <div className={`w-1 h-1 rounded-full ${streamError ? 'bg-red-400' : 'bg-green-400'}`} />
          {streamError ? 'Offline' : 'Connected'}
        </div>
      </div>
    </motion.div>
  )
}

export default function LivePage() {
  const [layout, setLayout] = useState<'grid' | 'list'>('grid')
  const [selectedFloor, setSelectedFloor] = useState<number | 'all'>('all')
  // Initialize with hardcoded cameras immediately
  const [cameras, setCameras] = useState<Camera[]>(() => {
    return HARDCODED_CAMERAS.map((cam) => ({
      id: cam.id,
      label: cam.label,
      streamUrl: cam.streamUrl,
      floor: cam.floor,
      active: cam.active,
      status: cam.status,
    }))
  })
  const [violations, setViolations] = useState<Violation[]>([])
  const [backendConnected, setBackendConnected] = useState(false)

  // Load cameras from Python backend
  useEffect(() => {
    const loadCameras = async () => {
      try {
        // First try the Python backend
        const response = await fetch(`${BACKEND_URL}/cameras`)

        if (response.ok) {
          const data = await response.json()
          setBackendConnected(true)

          if (data.cameras && Array.isArray(data.cameras) && data.cameras.length > 0) {
            // Show Factory001, Factory002, and violation videos only
            // Skip Factory003/004 (dark/empty videos) and test footage
            const factory001 = data.cameras.filter((cam: any) => {
              const label = cam.label.toLowerCase()
              return label.includes('factory001')
            })

            const factory002 = data.cameras.filter((cam: any) => {
              const label = cam.label.toLowerCase()
              return label.includes('factory002')
            })

            const violations = data.cameras.filter((cam: any) => {
              const label = cam.label.toLowerCase()
              return label.includes('violation')
            })

            // Combine: Factory001, Factory002 only (limit to 6 for 2x3 grid)
            const combined = [...factory001, ...factory002].slice(0, 6)

            const loadedCameras: Camera[] = combined.map((cam: any) => ({
              id: cam.id,
              label: cam.label,
              streamUrl: `${BACKEND_URL}/video_feed/${cam.id}`,
              floor: cam.floor || 1,
              active: cam.status === 'active',
              status: 'normal' as const,
            }))
            setCameras(loadedCameras)
            console.log(`✅ Loaded ${loadedCameras.length} cameras from Python backend (filtered from ${data.cameras.length})`)
          } else {
            console.log('⚠️ Backend returned no cameras, preserving existing hardcoded cameras.')
          }
        } else {
          console.warn('Python backend not available, falling back to user API')
          await loadFromUserAPI()
        }
      } catch (error) {
        console.warn('Backend fetch failed, falling back to user API:', error)
        await loadFromUserAPI()
      }
    }

    const loadFromUserAPI = async () => {
      try {
        const token = localStorage.getItem('token')
        if (!token) {
          console.warn('No auth token found')
          return
        }

        const response = await fetch('/api/user/data', {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        })

        if (!response.ok) {
          console.error('Failed to load cameras from API')
          return
        }

        const data = await response.json()

        if (data.cameras && Array.isArray(data.cameras) && data.cameras.length > 0) {
          const loadedCameras: Camera[] = data.cameras.map((cam: any) => ({
            id: cam.id || `cam-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            label: cam.label,
            streamUrl: cam.streamUrl,
            floor: cam.floor,
            active: cam.active !== false,
            status: cam.status || 'normal',
          }))
          setCameras(loadedCameras)
          console.log(`✅ Loaded ${loadedCameras.length} cameras from user API`)
        } else {
          console.log('⚠️ API returned no cameras, preserving existing hardcoded cameras.')
          // Ensure we have hardcoded cameras if API returns empty
          setCameras(prevCameras => {
            if (prevCameras.length === 0) {
              return HARDCODED_CAMERAS.map((cam) => ({
                id: cam.id,
                label: cam.label,
                streamUrl: cam.streamUrl,
                floor: cam.floor,
                active: cam.active,
                status: cam.status,
              }))
            }
            return prevCameras
          })
        }
      } catch (error) {
        console.error('Failed to load cameras:', error)
      }
    }

    loadCameras()
  }, [])

  // Poll for violations from Python backend
  useEffect(() => {
    if (!backendConnected) return

    const pollViolations = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/status`)
        if (response.ok) {
          const data = await response.json()
          setViolations(data.violations || [])
        }
      } catch (error) {
        // Silently fail - backend might be temporarily unavailable
      }
    }

    // Poll immediately
    pollViolations()

    // Then poll every second
    const interval = setInterval(pollViolations, 1000)
    return () => clearInterval(interval)
  }, [backendConnected])

  // Helper to check if a camera has violations
  const cameraHasViolation = (cameraId: string) => {
    return violations.some(v => v.camera_id === cameraId)
  }
  
  const filteredCameras = selectedFloor === 'all' 
    ? cameras 
    : cameras.filter(cam => cam.floor === selectedFloor)
  
  const gridCols = layout === 'grid' ? 'grid-cols-3' : 'grid-cols-1'
  
  return (
    <div className="flex-1 p-6 overflow-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-semibold text-white mb-1">Footage Review</h1>
          <div className="flex items-center gap-4">
            <p className="text-white/50 text-sm">
              {filteredCameras.length} feed{filteredCameras.length !== 1 ? 's' : ''} processing
            </p>
            {backendConnected && (
              <div className="flex items-center gap-1 text-xs text-green-400">
                <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                AI Backend Connected
              </div>
            )}
            {violations.length > 0 && (
              <div className="flex items-center gap-1 px-2 py-0.5 bg-red-500/20 border border-red-500/30 rounded text-xs text-red-400">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                  <line x1="12" y1="9" x2="12" y2="13" />
                  <line x1="12" y1="17" x2="12.01" y2="17" />
                </svg>
                {violations.length} Active Violation{violations.length !== 1 ? 's' : ''}
              </div>
            )}
          </div>
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
          {filteredCameras.map((camera) => (
            <CameraFeedCard
              key={camera.id}
              camera={camera}
              layout={layout}
              hasViolation={cameraHasViolation(camera.id)}
            />
          ))}
        </div>
      )}
    </div>
  )
}

