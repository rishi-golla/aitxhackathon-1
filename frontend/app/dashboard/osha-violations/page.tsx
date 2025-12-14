'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface Accident {
  id: string
  type: 'crowd' | 'smoke' | 'fall' | 'fire' | 'collision' | 'spill'
  severity: 'low' | 'medium' | 'high' | 'critical'
  title: string
  description: string
  cameraId: string
  cameraName: string
  floor: number
  timestamp: string
  duration: string
  videoUrl: string
  thumbnailUrl?: string
  status: 'new' | 'reviewing' | 'resolved' | 'archived'
  assignedTo?: string
}

const mockAccidents: Accident[] = [
  {
    id: 'acc-001',
    type: 'crowd',
    severity: 'high',
    title: 'Crowd Detected',
    description: 'Large gathering detected in restricted area. Potential safety hazard.',
    cameraId: 'cam-002',
    cameraName: 'Camera N°2',
    floor: 1,
    timestamp: '2024-01-15 14:32:18',
    duration: '00:04:23',
    videoUrl: '/videos/crowd-001.mp4',
    status: 'new',
  },
  {
    id: 'acc-002',
    type: 'smoke',
    severity: 'critical',
    title: 'Smoke Detected',
    description: 'Smoke detected in production area. Immediate attention required.',
    cameraId: 'cam-017',
    cameraName: 'Camera N°17',
    floor: 2,
    timestamp: '2024-01-15 13:15:42',
    duration: '00:02:15',
    videoUrl: '/videos/smoke-001.mp4',
    status: 'reviewing',
    assignedTo: 'Safety Team A',
  },
  {
    id: 'acc-003',
    type: 'fall',
    severity: 'critical',
    title: 'Person Fall Detected',
    description: 'Individual detected falling. Emergency response initiated.',
    cameraId: 'cam-012',
    cameraName: 'Camera N°12',
    floor: 1,
    timestamp: '2024-01-15 11:45:33',
    duration: '00:01:45',
    videoUrl: '/videos/fall-001.mp4',
    status: 'resolved',
    assignedTo: 'Medical Team',
  },
  {
    id: 'acc-004',
    type: 'fire',
    severity: 'critical',
    title: 'Fire Detected',
    description: 'Fire detected in storage area. Fire suppression system activated.',
    cameraId: 'cam-008',
    cameraName: 'Camera N°8',
    floor: 3,
    timestamp: '2024-01-15 09:22:11',
    duration: '00:08:56',
    videoUrl: '/videos/fire-001.mp4',
    status: 'resolved',
    assignedTo: 'Fire Safety Team',
  },
  {
    id: 'acc-005',
    type: 'collision',
    severity: 'medium',
    title: 'Vehicle Collision',
    description: 'Minor collision between two forklifts. No injuries reported.',
    cameraId: 'cam-021',
    cameraName: 'Camera N°21',
    floor: 1,
    timestamp: '2024-01-14 16:18:27',
    duration: '00:03:12',
    videoUrl: '/videos/collision-001.mp4',
    status: 'resolved',
  },
  {
    id: 'acc-006',
    type: 'spill',
    severity: 'low',
    title: 'Liquid Spill',
    description: 'Liquid spill detected on floor. Cleaning crew notified.',
    cameraId: 'cam-015',
    cameraName: 'Camera N°15',
    floor: 2,
    timestamp: '2024-01-14 14:52:08',
    duration: '00:01:33',
    videoUrl: '/videos/spill-001.mp4',
    status: 'archived',
  },
]

const FilterIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M2 4H14M4 8H12M6 12H10" strokeLinecap="round" />
  </svg>
)

const SearchIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
    <circle cx="7" cy="7" r="4" />
    <path d="M10 10L13 13" strokeLinecap="round" />
  </svg>
)

const PlayIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
    <polygon points="5 3 19 12 5 21 5 3" />
  </svg>
)

const DownloadIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M8 2V11M8 11L5 8M8 11L11 8" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M2 12V13C2 13.5523 2.44772 14 3 14H13C13.5523 14 14 13.5523 14 13V12" strokeLinecap="round" />
  </svg>
)

const MoreIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
    <circle cx="8" cy="3" r="1.5" />
    <circle cx="8" cy="8" r="1.5" />
    <circle cx="8" cy="13" r="1.5" />
  </svg>
)

const getTypeIcon = (type: string) => {
  const icons = {
    crowd: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="5" cy="4" r="2" />
        <circle cx="11" cy="4" r="2" />
        <path d="M2 13C2 11.3431 3.34315 10 5 10C6.65685 10 8 11.3431 8 13" />
        <path d="M8 13C8 11.3431 9.34315 10 11 10C12.6569 10 14 11.3431 14 13" />
      </svg>
    ),
    smoke: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M3 12C3 12 4 10 4 8C4 6 5 4 7 4C9 4 10 6 10 8" strokeLinecap="round" />
        <path d="M6 12C6 12 7 11 7 9.5C7 8 8 7 9 7C10 7 11 8 11 9.5" strokeLinecap="round" />
        <path d="M2 14H14" strokeLinecap="round" />
      </svg>
    ),
    fall: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="10" cy="3" r="2" />
        <path d="M4 14L8 9L6 7L10 5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
    fire: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M8 2C8 2 6 4 6 6C6 7 6.5 8 7.5 8.5C7 9 6.5 10 6.5 11C6.5 12.5 7.5 14 9.5 14C11.5 14 12.5 12.5 12.5 11C12.5 9 11 8 10.5 7C11 6 12 5 12 3.5C12 3.5 10 4.5 9 6C8.5 5 8 3 8 2Z" strokeLinejoin="round" />
      </svg>
    ),
    collision: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="5" cy="8" r="3" />
        <circle cx="11" cy="8" r="3" />
        <path d="M8 8L10 6M8 8L10 10" strokeLinecap="round" />
      </svg>
    ),
    spill: (
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M8 2V8M8 8L5 5M8 8L11 5" strokeLinecap="round" strokeLinejoin="round" />
        <ellipse cx="8" cy="12" rx="5" ry="2" />
      </svg>
    ),
  }
  return icons[type as keyof typeof icons] || icons.crowd
}

const getSeverityColor = (severity: string) => {
  const colors = {
    low: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
    medium: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
    high: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
    critical: 'bg-red-500/20 text-red-400 border-red-500/30',
  }
  return colors[severity as keyof typeof colors]
}

const getStatusColor = (status: string) => {
  const colors = {
    new: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30',
    reviewing: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
    resolved: 'bg-green-500/20 text-green-400 border-green-500/30',
    archived: 'bg-gray-500/20 text-gray-400 border-gray-500/30',
  }
  return colors[status as keyof typeof colors]
}

const AccidentCard = ({ accident }: { accident: Accident }) => {
  const [isExpanded, setIsExpanded] = useState(false)
  const [isHovered, setIsHovered] = useState(false)

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-xl bg-card border border-white/5 hover:border-cyan/20 transition-all duration-300 overflow-hidden"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div className="p-4">
        <div className="flex gap-4">
          {/* Video Thumbnail */}
          <div className="w-48 h-28 rounded-lg bg-zinc-900 relative overflow-hidden shrink-0 group cursor-pointer">
            <div className="absolute inset-0 flex items-center justify-center bg-black/50 group-hover:bg-black/30 transition-colors">
              <div className="w-12 h-12 rounded-full bg-cyan/20 flex items-center justify-center group-hover:bg-cyan/30 transition-colors">
                <PlayIcon />
              </div>
            </div>
            
            {/* Duration Badge */}
            <div className="absolute bottom-2 right-2 px-2 py-0.5 rounded bg-black/70 text-white text-[10px] font-medium backdrop-blur-sm">
              {accident.duration}
            </div>
            
            {/* Live Recording Badge */}
            <div className="absolute top-2 left-2 px-2 py-0.5 rounded bg-red-500 text-white text-[9px] font-bold flex items-center gap-1">
              <div className="w-1 h-1 rounded-full bg-white" />
              REC
            </div>
          </div>

          {/* Info */}
          <div className="flex-1 min-w-0">
            <div className="flex items-start justify-between mb-2">
              <div className="flex items-center gap-2">
                <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${getSeverityColor(accident.severity)}`}>
                  {getTypeIcon(accident.type)}
                </div>
                <div>
                  <h3 className="text-white font-semibold text-base">{accident.title}</h3>
                  <p className="text-white/50 text-xs">{accident.cameraName} • Floor {accident.floor}</p>
                </div>
              </div>
              
              <div className="flex items-center gap-2">
                <div className={`px-2 py-1 rounded text-[10px] font-medium border ${getSeverityColor(accident.severity)}`}>
                  {accident.severity.toUpperCase()}
                </div>
                <div className={`px-2 py-1 rounded text-[10px] font-medium border ${getStatusColor(accident.status)}`}>
                  {accident.status.charAt(0).toUpperCase() + accident.status.slice(1)}
                </div>
              </div>
            </div>

            <p className="text-white/70 text-sm mb-3 line-clamp-2">{accident.description}</p>

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4 text-xs text-white/40">
                <div className="flex items-center gap-1">
                  <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <circle cx="8" cy="8" r="6" />
                    <path d="M8 4V8L10 10" strokeLinecap="round" />
                  </svg>
                  {accident.timestamp}
                </div>
                {accident.assignedTo && (
                  <div className="flex items-center gap-1">
                    <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                      <circle cx="8" cy="6" r="3" />
                      <path d="M3 14C3 11.7909 4.79086 10 7 10H9C11.2091 10 13 11.7909 13 14" />
                    </svg>
                    {accident.assignedTo}
                  </div>
                )}
              </div>

              <div className="flex items-center gap-2">
                <button className="p-2 rounded-lg bg-white/5 hover:bg-cyan/10 text-white/70 hover:text-cyan transition-all">
                  <DownloadIcon />
                </button>
                <button 
                  onClick={() => setIsExpanded(!isExpanded)}
                  className="px-3 py-1.5 rounded-lg bg-cyan/10 hover:bg-cyan/20 text-cyan text-xs font-medium transition-all"
                >
                  {isExpanded ? 'Hide Details' : 'View Details'}
                </button>
                <button className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-white/70 hover:text-white transition-all">
                  <MoreIcon />
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Expanded Details */}
        <AnimatePresence>
          {isExpanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="mt-4 pt-4 border-t border-white/10"
            >
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <p className="text-white/50 text-xs mb-1">Incident Type</p>
                  <p className="text-white text-sm font-medium capitalize">{accident.type}</p>
                </div>
                <div>
                  <p className="text-white/50 text-xs mb-1">Camera ID</p>
                  <p className="text-white text-sm font-medium">{accident.cameraId}</p>
                </div>
                <div>
                  <p className="text-white/50 text-xs mb-1">Video URL</p>
                  <p className="text-cyan text-sm font-medium truncate">{accident.videoUrl}</p>
                </div>
              </div>
              
              <div className="mt-4 flex gap-2">
                <button className="flex-1 py-2 rounded-lg bg-green-500/10 border border-green-500/30 text-green-400 text-sm font-medium hover:bg-green-500/20 transition-all">
                  Mark as Resolved
                </button>
                <button className="flex-1 py-2 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-400 text-sm font-medium hover:bg-amber-500/20 transition-all">
                  Assign to Team
                </button>
                <button className="flex-1 py-2 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-sm font-medium hover:bg-red-500/20 transition-all">
                  Archive
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  )
}

export default function AccidentsPage() {
  const [searchQuery, setSearchQuery] = useState('')
  const [filterSeverity, setFilterSeverity] = useState<string>('all')
  const [filterStatus, setFilterStatus] = useState<string>('all')
  const [accidents] = useState<Accident[]>(mockAccidents)

  const filteredAccidents = accidents.filter(accident => {
    const matchesSearch = accident.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         accident.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         accident.cameraName.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesSeverity = filterSeverity === 'all' || accident.severity === filterSeverity
    const matchesStatus = filterStatus === 'all' || accident.status === filterStatus
    return matchesSearch && matchesSeverity && matchesStatus
  })

  const stats = {
    total: accidents.length,
    new: accidents.filter(a => a.status === 'new').length,
    critical: accidents.filter(a => a.severity === 'critical').length,
    resolved: accidents.filter(a => a.status === 'resolved').length,
  }

  return (
    <div className="flex-1 p-6 overflow-auto h-full">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-semibold text-white mb-1">OSHA Violations</h1>
        <p className="text-white/50 text-sm">
          {filteredAccidents.length} incident{filteredAccidents.length !== 1 ? 's' : ''} recorded
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <div className="rounded-xl bg-card border border-white/5 p-4">
          <p className="text-white/50 text-xs mb-1">Total Incidents</p>
          <p className="text-white text-2xl font-bold">{stats.total}</p>
        </div>
        <div className="rounded-xl bg-card border border-cyan-500/20 p-4">
          <p className="text-cyan/70 text-xs mb-1">New Reports</p>
          <p className="text-cyan text-2xl font-bold">{stats.new}</p>
        </div>
        <div className="rounded-xl bg-card border border-red-500/20 p-4">
          <p className="text-red-400/70 text-xs mb-1">Critical</p>
          <p className="text-red-400 text-2xl font-bold">{stats.critical}</p>
        </div>
        <div className="rounded-xl bg-card border border-green-500/20 p-4">
          <p className="text-green-400/70 text-xs mb-1">Resolved</p>
          <p className="text-green-400 text-2xl font-bold">{stats.resolved}</p>
        </div>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-3 mb-6">
        {/* Search */}
        <div className="flex-1 relative">
          <SearchIcon />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search incidents..."
            className="w-full pl-10 pr-4 py-2.5 bg-card border border-white/10 rounded-lg text-white placeholder:text-white/40 focus:outline-none focus:border-cyan/50 focus:ring-2 focus:ring-cyan/20 transition-all"
          />
          <div className="absolute left-3 top-1/2 -translate-y-1/2 text-white/40">
            <SearchIcon />
          </div>
        </div>

        {/* Severity Filter */}
        <div className="flex items-center gap-2 px-4 py-2 bg-card border border-white/10 rounded-lg">
          <FilterIcon />
          <span className="text-sm text-zinc-400">Severity:</span>
          {['all', 'low', 'medium', 'high', 'critical'].map((severity) => (
            <button
              key={severity}
              onClick={() => setFilterSeverity(severity)}
              className={`px-3 py-1 rounded-lg text-xs font-medium transition-all ${
                filterSeverity === severity
                  ? 'bg-cyan text-dark'
                  : 'text-zinc-400 hover:text-white hover:bg-white/5'
              }`}
            >
              {severity.charAt(0).toUpperCase() + severity.slice(1)}
            </button>
          ))}
        </div>

        {/* Status Filter */}
        <div className="flex items-center gap-2 px-4 py-2 bg-card border border-white/10 rounded-lg">
          <span className="text-sm text-zinc-400">Status:</span>
          {['all', 'new', 'reviewing', 'resolved', 'archived'].map((status) => (
            <button
              key={status}
              onClick={() => setFilterStatus(status)}
              className={`px-3 py-1 rounded-lg text-xs font-medium transition-all ${
                filterStatus === status
                  ? 'bg-cyan text-dark'
                  : 'text-zinc-400 hover:text-white hover:bg-white/5'
              }`}
            >
              {status.charAt(0).toUpperCase() + status.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Accidents List */}
      <div className="space-y-4">
        {filteredAccidents.map((accident) => (
          <AccidentCard key={accident.id} accident={accident} />
        ))}
      </div>

      {filteredAccidents.length === 0 && (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-white/5 flex items-center justify-center">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M12 3L19 17H5L12 3Z" strokeLinejoin="round" />
                <path d="M12 9V13" strokeLinecap="round" />
                <circle cx="12" cy="16" r="0.5" fill="currentColor" />
              </svg>
            </div>
            <h3 className="text-white text-lg font-medium mb-2">No Accidents Found</h3>
            <p className="text-white/50 text-sm">No incidents match your current filters.</p>
          </div>
        </div>
      )}
    </div>
  )
}

