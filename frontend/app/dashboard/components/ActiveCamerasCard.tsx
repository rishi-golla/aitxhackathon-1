'use client'

import { motion } from 'framer-motion'
import { useEffect, useState } from 'react'

const ChevronRightIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M6 4L10 8L6 12" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
)

export default function ActiveCamerasCard() {
  const [animated, setAnimated] = useState(false)
  const [count, setCount] = useState(0)
  
  // Line graph data points - smooth increasing curve
  const lineData = [
    { x: 0, y: 50 },
    { x: 50, y: 48 },
    { x: 100, y: 42 },
    { x: 150, y: 36 },
    { x: 200, y: 28 },
    { x: 250, y: 20 },
    { x: 300, y: 14 },
    { x: 350, y: 8 },
    { x: 400, y: 4 },
  ]
  
  // Generate smooth bezier curve path
  const generatePath = () => {
    let path = `M ${lineData[0].x} ${lineData[0].y}`
    for (let i = 1; i < lineData.length; i++) {
      const prev = lineData[i - 1]
      const curr = lineData[i]
      const cp1x = prev.x + (curr.x - prev.x) * 0.5
      const cp1y = prev.y
      const cp2x = curr.x - (curr.x - prev.x) * 0.5
      const cp2y = curr.y
      path += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${curr.x} ${curr.y}`
    }
    return path
  }
  
  const linePath = generatePath()
  const areaPath = `${linePath} L 400 60 L 0 60 Z`
  
  useEffect(() => {
    // Animate counter
    const duration = 1000
    const steps = 60
    const increment = 24 / steps
    let current = 0
    
    const timer = setInterval(() => {
      current += increment
      if (current >= 24) {
        setCount(24)
        clearInterval(timer)
      } else {
        setCount(Math.floor(current))
      }
    }, duration / steps)
    
    setAnimated(true)
    
    return () => clearInterval(timer)
  }, [])

  return (
    <motion.div
      whileHover={{ y: -4 }}
      transition={{ duration: 0.25, ease: 'easeOut' }}
      className="
        relative w-full
        rounded-2xl
        bg-gradient-to-br from-zinc-900/80 to-zinc-950/90
        border border-white/5
        p-6
        shadow-[0_20px_40px_-20px_rgba(0,0,0,0.8)]
        backdrop-blur-xl
        hover:border-white/10
        transition-colors duration-300
      "
    >
      {/* Subtle inner glow */}
      <div className="pointer-events-none absolute inset-0 rounded-2xl ring-1 ring-white/10" />

      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <span className="text-sm tracking-wide text-zinc-400 uppercase">
          Active cameras
        </span>
        <button className="w-5 h-5 flex items-center justify-center text-zinc-500 hover:text-zinc-400 transition-colors">
          <ChevronRightIcon />
        </button>
      </div>

      {/* Metric */}
      <div className="flex items-baseline gap-2 mb-8">
        <motion.span 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.4 }}
          className="text-5xl font-semibold text-white"
        >
          {count}
        </motion.span>
        <span className="text-xl font-medium text-zinc-400">
          / 28
        </span>
      </div>

      {/* Line Chart */}
      <div className="relative h-32">
        <svg 
          width="100%" 
          height="100%" 
          viewBox="0 0 400 60" 
          preserveAspectRatio="none" 
          className="overflow-visible"
        >
          <defs>
            <linearGradient id="areaFillActive" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#3ddbd9" stopOpacity="0.2"/>
              <stop offset="100%" stopColor="#3ddbd9" stopOpacity="0"/>
            </linearGradient>
            <filter id="lineGlowActive">
              <feGaussianBlur stdDeviation="1.5" result="coloredBlur"/>
              <feMerge>
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
          </defs>
          
          {/* Area fill */}
          <motion.path 
            d={areaPath}
            fill="url(#areaFillActive)"
            initial={{ opacity: 0 }}
            animate={{ opacity: animated ? 1 : 0 }}
            transition={{ duration: 0.8, ease: 'easeOut', delay: 0.2 }}
          />
          
          {/* Main line */}
          <motion.path 
            d={linePath}
            fill="none"
            stroke="#3ddbd9"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            filter="url(#lineGlowActive)"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: animated ? 1 : 0, opacity: animated ? 1 : 0 }}
            transition={{ duration: 1.2, ease: 'easeOut', delay: 0.3 }}
          />
          
          {/* End dot */}
          <motion.circle 
            cx="400"
            cy="4"
            r="3"
            fill="#3ddbd9"
            filter="url(#lineGlowActive)"
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: animated ? 1 : 0, scale: animated ? 1 : 0 }}
            transition={{ duration: 0.3, ease: 'easeOut', delay: 1.5 }}
          />
        </svg>
        
        {/* Good Signal Badge */}
        <motion.div 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: animated ? 1 : 0, y: animated ? 0 : 10 }}
          transition={{ duration: 0.4, ease: 'easeOut', delay: 1.2 }}
          className="absolute right-0 bottom-0 flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-zinc-800/50 border border-white/5 backdrop-blur-sm"
        >
          <div className="w-1.5 h-1.5 rounded-full bg-[#3ddbd9] shadow-[0_0_6px_rgba(61,219,217,0.6)]"></div>
          <span className="text-[10px] text-[#3ddbd9] font-medium">Good signal</span>
        </motion.div>
      </div>
    </motion.div>
  )
}
