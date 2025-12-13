'use client'

import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import dynamic from 'next/dynamic'
import { WallSegment, CameraNode, FloorPlanVectors, InteractionMode } from '../types/floorplan'
import { autoTraceWalls } from '../utils/autoTrace'
import CameraStatusLegend from './CameraStatusLegend'

const FactorySceneManual = dynamic(() => import('./FactorySceneManual'), { 
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center bg-zinc-950">
      <div className="text-zinc-400 text-sm">Loading 3D environment...</div>
    </div>
  )
})

// Local storage key
const STORAGE_KEY = 'factory-floorplan-vectors'

function loadVectors(): FloorPlanVectors {
  if (typeof window === 'undefined') return { walls: [], cameras: [], referenceImage: null }
  
  const stored = localStorage.getItem(STORAGE_KEY)
  if (stored) {
    try {
      return JSON.parse(stored)
    } catch {
      return { walls: [], cameras: [], referenceImage: null }
    }
  }
  return { walls: [], cameras: [], referenceImage: null }
}

function saveVectors(vectors: FloorPlanVectors) {
  if (typeof window === 'undefined') return
  localStorage.setItem(STORAGE_KEY, JSON.stringify(vectors))
}

function CameraModal({
  isOpen,
  onClose,
  onSave,
}: {
  isOpen: boolean
  onClose: () => void
  onSave: (data: { label: string; streamUrl: string; floor: number }) => void
}) {
  const [label, setLabel] = useState('')
  const [streamUrl, setStreamUrl] = useState('')
  const [floor, setFloor] = useState(1)

  const handleSave = () => {
    if (label && streamUrl) {
      onSave({ label, streamUrl, floor })
      setLabel('')
      setStreamUrl('')
      setFloor(1)
      onClose()
    }
  }

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40"
            onClick={onClose}
          />

          <motion.div
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9, y: 20 }}
            transition={{ duration: 0.2, ease: 'easeOut' }}
            className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-50 w-full max-w-md"
          >
            <div className="bg-gradient-to-br from-zinc-900/95 to-zinc-950/95 border border-white/10 rounded-2xl p-6 shadow-2xl backdrop-blur-xl">
              <h3 className="text-xl font-semibold text-white mb-6">Add Camera</h3>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-zinc-400 mb-2">Camera Name</label>
                  <input
                    type="text"
                    value={label}
                    onChange={(e) => setLabel(e.target.value)}
                    placeholder="e.g., Camera №1"
                    className="w-full px-4 py-2.5 bg-zinc-800/50 border border-white/10 rounded-lg text-white placeholder:text-zinc-500 focus:outline-none focus:border-cyan-400/50 focus:ring-2 focus:ring-cyan-400/20 transition-all"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-zinc-400 mb-2">Stream URL</label>
                  <input
                    type="text"
                    value={streamUrl}
                    onChange={(e) => setStreamUrl(e.target.value)}
                    placeholder="rtsp://... or https://..."
                    className="w-full px-4 py-2.5 bg-zinc-800/50 border border-white/10 rounded-lg text-white placeholder:text-zinc-500 focus:outline-none focus:border-cyan-400/50 focus:ring-2 focus:ring-cyan-400/20 transition-all"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-zinc-400 mb-2">Floor Number</label>
                  <select
                    value={floor}
                    onChange={(e) => setFloor(Number(e.target.value))}
                    className="w-full px-4 py-2.5 bg-zinc-800/50 border border-white/10 rounded-lg text-white focus:outline-none focus:border-cyan-400/50 focus:ring-2 focus:ring-cyan-400/20 transition-all"
                  >
                    {[1, 2, 3, 4, 5].map((f) => (
                      <option key={f} value={f}>Floor {f}</option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="flex gap-3 mt-6">
                <button
                  onClick={onClose}
                  className="flex-1 px-4 py-2.5 bg-zinc-800/50 border border-white/10 rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all"
                >
                  Cancel
                </button>
                <button
                  onClick={handleSave}
                  disabled={!label || !streamUrl}
                  className="flex-1 px-4 py-2.5 bg-cyan-400 rounded-lg text-zinc-900 font-medium hover:bg-cyan-300 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                >
                  Save Camera
                </button>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}

function CameraViewer({ camera, onClose }: { camera: CameraNode | null; onClose: () => void }) {
  return (
    <AnimatePresence>
      {camera && (
        <motion.div
          initial={{ opacity: 0, x: 400 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: 400 }}
          transition={{ duration: 0.3, ease: 'easeOut' }}
          className="fixed right-6 top-24 z-50 w-96 bg-gradient-to-br from-zinc-900/95 to-zinc-950/95 border border-white/10 rounded-2xl shadow-2xl backdrop-blur-xl overflow-hidden"
        >
          <div className="flex items-center justify-between p-4 border-b border-white/10">
            <div>
              <h3 className="text-lg font-semibold text-white">{camera.label}</h3>
              <p className="text-xs text-zinc-400">Floor {camera.floor}</p>
            </div>
            <button
              onClick={onClose}
              className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-white/10 transition-colors text-zinc-400"
            >
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M4 4L12 12M12 4L4 12" strokeLinecap="round" />
              </svg>
            </button>
          </div>

          <div className="relative aspect-video bg-zinc-950">
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <div className="w-16 h-16 mx-auto mb-3 rounded-full bg-cyan-400/10 flex items-center justify-center">
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#3ddbd9" strokeWidth="2">
                    <path d="M23 7L16 12L23 17V7Z" />
                    <rect x="1" y="5" width="15" height="14" rx="2" />
                  </svg>
                </div>
                <p className="text-sm text-zinc-400">Camera Stream</p>
                <p className="text-xs text-zinc-500 mt-1 px-4 truncate">{camera.streamUrl}</p>
              </div>
            </div>
          </div>

          <div className="p-4 space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-zinc-400">Status</span>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-green-400 shadow-[0_0_8px_rgba(74,222,128,0.6)]" />
                <span className="text-green-400 font-medium">Live</span>
              </div>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-zinc-400">Position</span>
              <span className="text-white font-mono text-xs">
                ({camera.position[0].toFixed(1)}, {camera.position[2].toFixed(1)})
              </span>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

export default function DigitalTwinManual() {
  const [vectors, setVectors] = useState<FloorPlanVectors>({ walls: [], cameras: [], referenceImage: null })
  const [currentFloor, setCurrentFloor] = useState(1)
  const [mode, setMode] = useState<InteractionMode>('view')
  const [selectedCamera, setSelectedCamera] = useState<string | null>(null)
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [pendingPosition, setPendingPosition] = useState<[number, number, number] | null>(null)
  const [isAutoTracing, setIsAutoTracing] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    setVectors(loadVectors())
  }, [])

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = async (event) => {
        const imageUrl = event.target?.result as string
        const newVectors = { ...vectors, referenceImage: imageUrl }
        setVectors(newVectors)
        saveVectors(newVectors)
        
        // Automatically trace walls
        setIsAutoTracing(true)
        try {
          const detectedWalls = await autoTraceWalls(imageUrl, currentFloor)
          
          if (detectedWalls.length > 0) {
            const updatedVectors = {
              ...newVectors,
              walls: [...newVectors.walls, ...detectedWalls],
            }
            setVectors(updatedVectors)
            saveVectors(updatedVectors)
          }
        } catch (error) {
          console.error('Auto-trace error:', error)
        } finally {
          setIsAutoTracing(false)
        }
      }
      reader.readAsDataURL(file)
    }
  }

  const handleCameraPlace = (position: [number, number, number]) => {
    console.log('📍 Camera placement at:', position)
    setPendingPosition(position)
    setIsModalOpen(true)
  }

  const handleSaveCamera = (data: { label: string; streamUrl: string; floor: number }) => {
    if (pendingPosition) {
      const newCamera: CameraNode = {
        id: `camera-${Date.now()}`,
        position: pendingPosition,
        rotation: [0, 0, 0],
        ...data,
        active: true,
      }
      const updatedVectors = {
        ...vectors,
        cameras: [...vectors.cameras, newCamera],
      }
      setVectors(updatedVectors)
      saveVectors(updatedVectors)
      setPendingPosition(null)
    }
  }

  const handleClearWalls = () => {
    if (confirm('Clear all walls?')) {
      const updatedVectors = { ...vectors, walls: [] }
      setVectors(updatedVectors)
      saveVectors(updatedVectors)
    }
  }


  const selectedCameraData = vectors.cameras.find((cam) => cam.id === selectedCamera) || null

  return (
    <div className="relative w-full h-full">
      <div 
        className={`absolute inset-0 ${
          mode === 'place-cameras' ? 'cursor-crosshair' : ''
        }`}
      >
        <FactorySceneManual
          referenceImage={vectors.referenceImage}
          walls={vectors.walls}
          cameras={vectors.cameras}
          currentFloor={currentFloor}
          mode={mode}
          selectedCamera={selectedCamera}
          onCameraClick={(id) => setSelectedCamera(id === selectedCamera ? null : id)}
          onCameraPlace={handleCameraPlace}
        />
      </div>

      <div className="absolute inset-0 pointer-events-none">
        {/* Camera Status Legend */}
        <CameraStatusLegend />

        <div className="pointer-events-auto p-6">
          <div className="flex items-center gap-4 flex-wrap">
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isAutoTracing}
              className="px-4 py-2.5 bg-zinc-900/90 border border-white/10 rounded-lg text-white hover:bg-zinc-800 transition-all backdrop-blur-xl shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <div className="flex items-center gap-2">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M8 12V4M8 4L5 7M8 4L11 7" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M2 12V13C2 13.5523 2.44772 14 3 14H13C13.5523 14 14 13.5523 14 13V12" strokeLinecap="round" />
                </svg>
                <span className="text-sm font-medium">
                  {vectors.referenceImage ? 'Change Reference' : 'Upload Floor Plan'}
                </span>
              </div>
            </button>
            <input ref={fileInputRef} type="file" accept="image/*" onChange={handleImageUpload} className="hidden" />

            <div className="flex items-center gap-2 px-4 py-2.5 bg-zinc-900/90 border border-white/10 rounded-lg backdrop-blur-xl shadow-lg">
              <span className="text-sm text-zinc-400">Floor:</span>
              {[1, 2, 3, 4, 5].map((floor) => (
                <button
                  key={floor}
                  onClick={() => setCurrentFloor(floor)}
                  className={`w-8 h-8 rounded-lg text-sm font-medium transition-all ${
                    currentFloor === floor
                      ? 'bg-cyan-400 text-zinc-900 shadow-[0_0_12px_rgba(61,219,217,0.4)]'
                      : 'text-zinc-400 hover:text-white hover:bg-white/5'
                  }`}
                >
                  {floor}
                </button>
              ))}
            </div>

            {isAutoTracing && (
              <div className="px-4 py-2.5 bg-gradient-to-r from-cyan-500/20 to-blue-500/20 border border-cyan-400/50 rounded-lg backdrop-blur-xl shadow-lg">
                <div className="flex items-center gap-2">
                  <svg className="animate-spin" width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="8" cy="8" r="6" strokeOpacity="0.25" className="text-cyan-400" />
                    <path d="M8 2a6 6 0 0 1 6 6" strokeLinecap="round" className="text-cyan-400" />
                  </svg>
                  <span className="text-sm font-medium text-cyan-400">Auto-Tracing Walls...</span>
                </div>
              </div>
            )}

            <div className="flex items-center gap-2">
              <button
                onClick={() => {
                  if (confirm('Clear all cameras? This cannot be undone.')) {
                    const updatedVectors = {
                      ...vectors,
                      cameras: [],
                    }
                    setVectors(updatedVectors)
                    saveVectors(updatedVectors)
                    setSelectedCamera(null)
                  }
                }}
                disabled={vectors.cameras.length === 0}
                className="px-4 py-2.5 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm font-medium hover:bg-red-500/20 transition-all backdrop-blur-xl shadow-lg disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <div className="flex items-center gap-2">
                  <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M3 4H13M5 4V3C5 2.44772 5.44772 2 6 2H10C10.5523 2 11 2.44772 11 3V4M12 4V13C12 13.5523 11.5523 14 11 14H5C4.44772 14 4 13.5523 4 13V4" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  <span>Clear Cameras</span>
                </div>
              </button>

              <button
                onClick={() => setMode(mode === 'place-cameras' ? 'view' : 'place-cameras')}
                disabled={isAutoTracing}
                className={`px-4 py-2.5 border rounded-lg text-sm font-medium transition-all backdrop-blur-xl shadow-lg ${
                  mode === 'place-cameras'
                    ? 'bg-cyan-400 text-zinc-900 border-cyan-400'
                    : 'bg-zinc-900/90 text-white border-white/10 hover:bg-zinc-800'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                {mode === 'place-cameras' ? '✓ Placing Cameras' : 'Place Cameras'}
              </button>

              {vectors.walls.length > 0 && (
                <button
                  onClick={handleClearWalls}
                  disabled={isAutoTracing}
                  className="px-4 py-2.5 bg-zinc-900/90 border border-red-400/30 rounded-lg text-red-400 text-sm font-medium hover:bg-red-400/10 transition-all backdrop-blur-xl shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Clear Walls
                </button>
              )}
            </div>

            <div className="px-4 py-2.5 bg-zinc-900/90 border border-white/10 rounded-lg backdrop-blur-xl shadow-lg">
              <div className="flex items-center gap-4 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(61,219,217,0.6)]" />
                  <span className="text-white font-medium">
                    {vectors.walls.filter((w) => w.floor === currentFloor).length} Walls
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(61,219,217,0.6)]" />
                  <span className="text-white font-medium">
                    {vectors.cameras.filter((c) => c.floor === currentFloor).length} Cameras
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {mode === 'place-cameras' && (
          <div className="pointer-events-auto absolute bottom-6 left-1/2 -translate-x-1/2">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="px-6 py-3 bg-zinc-900/90 border border-cyan-400/30 rounded-lg backdrop-blur-xl shadow-lg"
            >
              <p className="text-sm text-cyan-400">
                Click anywhere on the floor to place a camera
              </p>
            </motion.div>
          </div>
        )}
      </div>

      <CameraModal
        isOpen={isModalOpen}
        onClose={() => {
          setIsModalOpen(false)
          setPendingPosition(null)
        }}
        onSave={handleSaveCamera}
      />

      <CameraViewer camera={selectedCameraData} onClose={() => setSelectedCamera(null)} />
    </div>
  )
}

