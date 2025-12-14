'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const BACKEND_URL = 'http://localhost:8000'

interface Violation {
  id: string
  videoFile: string
  title: string
  description: string
  oshaCode: string
  oshaTitle: string
  legalText: string
  penalty: number
  severity: 'low' | 'medium' | 'high' | 'critical'
  status: 'new' | 'reviewing' | 'resolved' | 'archived'
  detectedAt: string
  cameraId: string
  floor: number
  triggers: string[]
  duration: string
  summary?: string
}

// Violations data - loaded from config/violations.json structure
const violationsData: Violation[] = [
  {
    id: "vio-001",
    videoFile: "violation_1_bare_hands_industrial_machine_primary_violation.mp4",
    title: "Bare Hands Near Industrial Machine",
    description: "Worker operating industrial machinery without proper hand protection. Hands exposed to moving parts and potential crush hazards.",
    oshaCode: "29 CFR 1910.138(a)",
    oshaTitle: "General Requirements - Hand Protection",
    legalText: "Employers shall select and require employees to use appropriate hand protection when employees' hands are exposed to hazards such as those from skin absorption of harmful substances; severe cuts or lacerations; severe abrasions; punctures; chemical burns; thermal burns; and harmful temperature extremes.",
    penalty: 16131,
    severity: "critical",
    status: "new",
    detectedAt: "2024-12-14T10:23:45Z",
    cameraId: "cam-01",
    floor: 1,
    triggers: ["bare_hand", "industrial_machine"],
    duration: "00:02:15"
  },
  {
    id: "vio-002",
    videoFile: "violation_2_no_safety_glasses_sparks_incorrect.mp4",
    title: "No Safety Glasses - Sparks Hazard",
    description: "Worker exposed to flying sparks without proper eye protection. High risk of eye injury from hot metal particles.",
    oshaCode: "29 CFR 1910.133(a)(1)",
    oshaTitle: "Eye and Face Protection",
    legalText: "The employer shall ensure that each affected employee uses appropriate eye or face protection when exposed to eye or face hazards from flying particles, molten metal, liquid chemicals, acids or caustic liquids, chemical gases or vapors, or potentially injurious light radiation.",
    penalty: 16131,
    severity: "critical",
    status: "reviewing",
    detectedAt: "2024-12-14T09:15:22Z",
    cameraId: "cam-02",
    floor: 1,
    triggers: ["no_safety_glasses", "sparks"],
    duration: "00:01:48"
  },
  {
    id: "vio-003",
    videoFile: "violation_3_hand_near_spinning_blade.mp4",
    title: "Hand Near Spinning Blade",
    description: "Worker's hand detected in dangerous proximity to active spinning blade. Point of operation guarding violation.",
    oshaCode: "29 CFR 1910.212(a)(3)(ii)",
    oshaTitle: "Point of Operation Guarding",
    legalText: "The point of operation of machines whose operation exposes an employee to injury, shall be guarded. The guarding device shall be so designed and constructed as to prevent the operator from having any part of his body in the danger zone during the operating cycle.",
    penalty: 16131,
    severity: "critical",
    status: "new",
    detectedAt: "2024-12-14T11:42:18Z",
    cameraId: "cam-03",
    floor: 2,
    triggers: ["hand", "spinning_blade"],
    duration: "00:00:52"
  },
  {
    id: "vio-004",
    videoFile: "violation_4_unguarded_rotating_machinery.mp4",
    title: "Unguarded Rotating Machinery",
    description: "Rotating machinery operating without proper machine guarding. Exposed rotating parts present entanglement hazard.",
    oshaCode: "29 CFR 1910.212(a)(1)",
    oshaTitle: "Machine Guarding - General",
    legalText: "One or more methods of machine guarding shall be provided to protect the operator and other employees in the machine area from hazards such as those created by point of operation, ingoing nip points, rotating parts, flying chips and sparks.",
    penalty: 16131,
    severity: "high",
    status: "reviewing",
    detectedAt: "2024-12-14T08:33:11Z",
    cameraId: "cam-04",
    floor: 2,
    triggers: ["unguarded_machinery", "rotating_parts"],
    duration: "00:01:22"
  },
  {
    id: "vio-005",
    videoFile: "violation_5_correct_ppe_negative_example.mp4",
    title: "Correct PPE Usage (No Violation)",
    description: "Worker demonstrating proper PPE usage. Safety glasses, gloves, and hearing protection properly worn. Reference example for compliance.",
    oshaCode: "N/A",
    oshaTitle: "Compliant - No Violation",
    legalText: "This footage shows proper compliance with OSHA PPE requirements. No corrective action required.",
    penalty: 0,
    severity: "low",
    status: "resolved",
    detectedAt: "2024-12-14T07:15:00Z",
    cameraId: "cam-05",
    floor: 1,
    triggers: [],
    duration: "00:00:45"
  },
  {
    id: "vio-006",
    videoFile: "violation_6_chemical_handling_without_gloves.mp4",
    title: "Chemical Handling Without Gloves",
    description: "Worker handling chemical containers without appropriate chemical-resistant gloves. Skin exposure risk to hazardous substances.",
    oshaCode: "29 CFR 1910.138(a)",
    oshaTitle: "Hand Protection - Chemical Hazards",
    legalText: "Employers shall select and require employees to use appropriate hand protection when employees' hands are exposed to hazards such as those from skin absorption of harmful substances; severe cuts or lacerations; severe abrasions; punctures; chemical burns; thermal burns; and harmful temperature extremes.",
    penalty: 16131,
    severity: "high",
    status: "new",
    detectedAt: "2024-12-14T12:08:33Z",
    cameraId: "cam-06",
    floor: 3,
    triggers: ["bare_hand", "chemical_container"],
    duration: "00:01:05"
  },
  {
    id: "vio-007",
    videoFile: "violation_7_assembly_line_close_call.mp4",
    title: "Assembly Line Close Call",
    description: "Near-miss incident on assembly line. Worker narrowly avoided contact with moving machinery. Requires safety review.",
    oshaCode: "29 CFR 1910.212(a)(1)",
    oshaTitle: "Machine Guarding - General",
    legalText: "One or more methods of machine guarding shall be provided to protect the operator and other employees in the machine area from hazards such as those created by point of operation, ingoing nip points, rotating parts, flying chips and sparks.",
    penalty: 16131,
    severity: "medium",
    status: "reviewing",
    detectedAt: "2024-12-14T13:22:47Z",
    cameraId: "cam-07",
    floor: 1,
    triggers: ["close_call", "assembly_line"],
    duration: "00:00:38"
  }
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

const DGXBadge = () => (
  <div className="flex items-center gap-1 px-1.5 py-0.5 bg-gradient-to-r from-green-500/20 to-cyan-500/20 border border-green-500/30 rounded text-[8px] font-bold text-green-400">
    <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke="currentColor" strokeWidth="2" fill="none"/>
    </svg>
    DGX SPARK
  </div>
)

const DollarIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M8 1V15M11 4H6.5C5.67157 4 5 4.67157 5 5.5V5.5C5 6.32843 5.67157 7 6.5 7H9.5C10.3284 7 11 7.67157 11 8.5V8.5C11 9.32843 10.3284 10 9.5 10H5" strokeLinecap="round" />
  </svg>
)

const getSeverityColor = (severity: string) => {
  const colors = {
    low: 'bg-green-500/20 text-green-400 border-green-500/30',
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

const formatCurrency = (amount: number) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(amount)
}

const formatDate = (dateString: string) => {
  return new Date(dateString).toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

const ViolationCard = ({ violation }: { violation: Violation }) => {
  const [isExpanded, setIsExpanded] = useState(false)
  const [summary, setSummary] = useState<string>('Loading AI analysis...')
  const [summaryLoading, setSummaryLoading] = useState(true)

  // Fetch AI summary from backend
  useEffect(() => {
    const fetchSummary = async () => {
      try {
        // Find the camera ID that matches this violation video
        const response = await fetch(`${BACKEND_URL}/summarize/${violation.cameraId}`)
        if (response.ok) {
          const data = await response.json()
          setSummary(data.summary || violation.description)
        } else {
          setSummary(violation.description)
        }
      } catch {
        // Fallback to description if backend unavailable
        setSummary(violation.description)
      } finally {
        setSummaryLoading(false)
      }
    }

    fetchSummary()
  }, [violation.cameraId, violation.description])

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`rounded-xl bg-card border transition-all duration-300 overflow-hidden ${
        violation.severity === 'critical'
          ? 'border-red-500/30 shadow-lg shadow-red-500/10'
          : 'border-white/5 hover:border-cyan/20'
      }`}
    >
      <div className="p-4">
        <div className="flex gap-4">
          {/* Video Thumbnail */}
          <div className="w-56 h-32 rounded-lg bg-zinc-900 relative overflow-hidden shrink-0 group cursor-pointer">
            {/* MJPEG Stream from backend */}
            <img
              src={`${BACKEND_URL}/video_feed/${violation.cameraId}`}
              alt={violation.title}
              className="w-full h-full object-cover"
              onError={(e) => {
                (e.target as HTMLImageElement).style.display = 'none'
              }}
            />
            <div className="absolute inset-0 flex items-center justify-center bg-black/40 group-hover:bg-black/20 transition-colors">
              <div className="w-12 h-12 rounded-full bg-cyan/20 flex items-center justify-center group-hover:bg-cyan/30 transition-colors">
                <PlayIcon />
              </div>
            </div>

            {/* Duration Badge */}
            <div className="absolute bottom-2 right-2 px-2 py-0.5 rounded bg-black/70 text-white text-[10px] font-medium backdrop-blur-sm">
              {violation.duration}
            </div>

            {/* Violation Alert Badge */}
            {violation.penalty > 0 && (
              <div className="absolute top-2 left-2 px-2 py-0.5 rounded bg-red-500 text-white text-[9px] font-bold flex items-center gap-1">
                <div className="w-1 h-1 rounded-full bg-white animate-pulse" />
                VIOLATION
              </div>
            )}
          </div>

          {/* Info */}
          <div className="flex-1 min-w-0">
            <div className="flex items-start justify-between mb-2">
              <div className="flex items-center gap-3">
                <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${getSeverityColor(violation.severity)}`}>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 3L19 17H5L12 3Z" strokeLinejoin="round" />
                    <path d="M12 9V13" strokeLinecap="round" />
                    <circle cx="12" cy="16" r="0.5" fill="currentColor" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-white font-semibold text-base">{violation.title}</h3>
                  <p className="text-white/50 text-xs">{violation.oshaCode} • Floor {violation.floor}</p>
                </div>
              </div>

              <div className="flex items-center gap-2">
                {/* Fine Amount */}
                {violation.penalty > 0 && (
                  <div className="px-3 py-1.5 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-sm font-bold flex items-center gap-1">
                    <DollarIcon />
                    {formatCurrency(violation.penalty)}
                  </div>
                )}
                <div className={`px-2 py-1 rounded text-[10px] font-medium border ${getSeverityColor(violation.severity)}`}>
                  {violation.severity.toUpperCase()}
                </div>
                <div className={`px-2 py-1 rounded text-[10px] font-medium border ${getStatusColor(violation.status)}`}>
                  {violation.status.charAt(0).toUpperCase() + violation.status.slice(1)}
                </div>
              </div>
            </div>

            {/* AI Summary Section */}
            <div className="mb-3 p-2.5 rounded-lg bg-gradient-to-br from-zinc-800/50 to-zinc-900/50 border border-white/5">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[10px] font-semibold text-cyan uppercase tracking-wider">AI Video Analysis</span>
                <DGXBadge />
              </div>
              {summaryLoading ? (
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-cyan animate-pulse" />
                  <p className="text-xs text-white/50 italic">Analyzing with DGX Spark...</p>
                </div>
              ) : (
                <p className="text-xs text-white/70 leading-relaxed">{summary}</p>
              )}
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4 text-xs text-white/40">
                <div className="flex items-center gap-1">
                  <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <circle cx="8" cy="8" r="6" />
                    <path d="M8 4V8L10 10" strokeLinecap="round" />
                  </svg>
                  {formatDate(violation.detectedAt)}
                </div>
                <div className="flex items-center gap-1">
                  <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <rect x="2" y="3" width="12" height="10" rx="1" />
                    <circle cx="8" cy="8" r="2" />
                  </svg>
                  {violation.cameraId}
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button className="p-2 rounded-lg bg-white/5 hover:bg-cyan/10 text-white/70 hover:text-cyan transition-all">
                  <DownloadIcon />
                </button>
                <button
                  onClick={() => setIsExpanded(!isExpanded)}
                  className="px-3 py-1.5 rounded-lg bg-cyan/10 hover:bg-cyan/20 text-cyan text-xs font-medium transition-all"
                >
                  {isExpanded ? 'Hide Details' : 'View OSHA Details'}
                </button>
                <button className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-white/70 hover:text-white transition-all">
                  <MoreIcon />
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Expanded OSHA Details */}
        <AnimatePresence>
          {isExpanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="mt-4 pt-4 border-t border-white/10"
            >
              {/* OSHA Citation Box */}
              <div className="mb-4 p-4 rounded-lg bg-red-500/5 border border-red-500/20">
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 rounded-lg bg-red-500/20 flex items-center justify-center text-red-400 shrink-0">
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                      <rect x="2" y="2" width="12" height="12" rx="1" />
                      <path d="M5 5H11M5 8H11M5 11H8" strokeLinecap="round" />
                    </svg>
                  </div>
                  <div>
                    <p className="text-red-400 font-semibold text-sm mb-1">{violation.oshaCode}</p>
                    <p className="text-white font-medium text-sm mb-2">{violation.oshaTitle}</p>
                    <p className="text-white/60 text-xs leading-relaxed">{violation.legalText}</p>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-4 gap-4 mb-4">
                <div>
                  <p className="text-white/50 text-xs mb-1">Video File</p>
                  <p className="text-white text-sm font-medium truncate">{violation.videoFile}</p>
                </div>
                <div>
                  <p className="text-white/50 text-xs mb-1">Camera ID</p>
                  <p className="text-white text-sm font-medium">{violation.cameraId}</p>
                </div>
                <div>
                  <p className="text-white/50 text-xs mb-1">Detection Triggers</p>
                  <div className="flex flex-wrap gap-1">
                    {violation.triggers.length > 0 ? violation.triggers.map((trigger, i) => (
                      <span key={i} className="px-2 py-0.5 rounded bg-white/10 text-white/70 text-[10px]">
                        {trigger}
                      </span>
                    )) : (
                      <span className="text-white/50 text-sm">None</span>
                    )}
                  </div>
                </div>
                <div>
                  <p className="text-white/50 text-xs mb-1">Potential Fine</p>
                  <p className={`text-lg font-bold ${violation.penalty > 0 ? 'text-red-400' : 'text-green-400'}`}>
                    {violation.penalty > 0 ? formatCurrency(violation.penalty) : 'No Fine'}
                  </p>
                </div>
              </div>

              <div className="flex gap-2">
                <button className="flex-1 py-2 rounded-lg bg-green-500/10 border border-green-500/30 text-green-400 text-sm font-medium hover:bg-green-500/20 transition-all">
                  Mark as Resolved
                </button>
                <button className="flex-1 py-2 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-400 text-sm font-medium hover:bg-amber-500/20 transition-all">
                  Assign to Safety Team
                </button>
                <button className="flex-1 py-2 rounded-lg bg-cyan/10 border border-cyan/30 text-cyan text-sm font-medium hover:bg-cyan/20 transition-all">
                  Generate Report
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  )
}

export default function ViolationsPage() {
  const [searchQuery, setSearchQuery] = useState('')
  const [filterSeverity, setFilterSeverity] = useState<string>('all')
  const [filterStatus, setFilterStatus] = useState<string>('all')
  const [violations] = useState<Violation[]>(violationsData)

  const filteredViolations = violations.filter(violation => {
    const matchesSearch = violation.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         violation.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         violation.oshaCode.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesSeverity = filterSeverity === 'all' || violation.severity === filterSeverity
    const matchesStatus = filterStatus === 'all' || violation.status === filterStatus
    return matchesSearch && matchesSeverity && matchesStatus
  })

  const stats = {
    total: violations.length,
    totalFines: violations.reduce((sum, v) => sum + v.penalty, 0),
    critical: violations.filter(v => v.severity === 'critical').length,
    new: violations.filter(v => v.status === 'new').length,
  }

  return (
    <div className="flex-1 p-6 overflow-auto h-full">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-semibold text-white mb-1">OSHA Violations</h1>
        <p className="text-white/50 text-sm">
          {filteredViolations.length} violation{filteredViolations.length !== 1 ? 's' : ''} detected from video analysis
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <div className="rounded-xl bg-card border border-white/5 p-4">
          <p className="text-white/50 text-xs mb-1">Total Violations</p>
          <p className="text-white text-2xl font-bold">{stats.total}</p>
        </div>
        <div className="rounded-xl bg-card border border-red-500/20 p-4">
          <p className="text-red-400/70 text-xs mb-1">Total Potential Fines</p>
          <p className="text-red-400 text-2xl font-bold">{formatCurrency(stats.totalFines)}</p>
        </div>
        <div className="rounded-xl bg-card border border-orange-500/20 p-4">
          <p className="text-orange-400/70 text-xs mb-1">Critical Violations</p>
          <p className="text-orange-400 text-2xl font-bold">{stats.critical}</p>
        </div>
        <div className="rounded-xl bg-card border border-cyan-500/20 p-4">
          <p className="text-cyan/70 text-xs mb-1">New Reports</p>
          <p className="text-cyan text-2xl font-bold">{stats.new}</p>
        </div>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-3 mb-6">
        {/* Search */}
        <div className="flex-1 relative">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search violations by title, OSHA code..."
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
          {['all', 'new', 'reviewing', 'resolved'].map((status) => (
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

      {/* Violations List */}
      <div className="space-y-4">
        {filteredViolations.map((violation) => (
          <ViolationCard key={violation.id} violation={violation} />
        ))}
      </div>

      {filteredViolations.length === 0 && (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-white/5 flex items-center justify-center">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M12 3L19 17H5L12 3Z" strokeLinejoin="round" />
                <path d="M12 9V13" strokeLinecap="round" />
                <circle cx="12" cy="16" r="0.5" fill="currentColor" />
              </svg>
            </div>
            <h3 className="text-white text-lg font-medium mb-2">No Violations Found</h3>
            <p className="text-white/50 text-sm">No violations match your current filters.</p>
          </div>
        </div>
      )}
    </div>
  )
}
