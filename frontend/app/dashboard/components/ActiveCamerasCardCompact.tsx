'use client'

import { motion } from 'framer-motion'
import { useEffect, useState } from 'react'

export default function ActiveCamerasCardCompact() {
  const [animated, setAnimated] = useState(false)
  const [count, setCount] = useState(0)
  
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
      whileHover={{ y: -2 }}
      transition={{ duration: 0.25, ease: 'easeOut' }}
      className="relative w-full h-full rounded-xl bg-gradient-to-br from-zinc-900/80 to-zinc-950/90 border border-white/5 p-4 shadow-[0_20px_40px_-20px_rgba(0,0,0,0.8)] backdrop-blur-xl hover:border-white/10 transition-colors duration-300 flex flex-col"
    >
      <div className="pointer-events-none absolute inset-0 rounded-xl ring-1 ring-white/10" />

      <div className="flex items-center justify-between mb-2">
        <span className="text-xs tracking-wide text-zinc-400 uppercase">Active cameras</span>
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-zinc-500">
          <path d="M6 4L10 8L6 12" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </div>

      <div className="flex items-baseline gap-1.5 mb-3">
        <motion.span 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.4 }}
          className="text-3xl font-semibold text-white"
        >
          {count}
        </motion.span>
        <span className="text-sm text-zinc-400 font-medium">/ 28</span>
      </div>
      
      <div className="relative flex-1">
        <svg width="100%" height="100%" viewBox="0 0 400 60" preserveAspectRatio="none" className="overflow-visible">
          <defs>
            <linearGradient id="areaFillCamerasCompact" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#3ddbd9" stopOpacity="0.2"/>
              <stop offset="100%" stopColor="#3ddbd9" stopOpacity="0"/>
            </linearGradient>
            <filter id="lineGlowCompact">
              <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
              <feMerge>
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
          </defs>
          
          <motion.path 
            d={areaPath} 
            fill="url(#areaFillCamerasCompact)"
            initial={{ opacity: 0 }}
            animate={{ opacity: animated ? 1 : 0 }}
            transition={{ duration: 1, ease: 'easeOut', delay: 0.5 }}
          />
          
          <motion.path 
            d={linePath} 
            fill="none" 
            stroke="#3ddbd9" 
            strokeWidth="2.5" 
            strokeLinecap="round"
            filter="url(#lineGlowCompact)"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: animated ? 1 : 0 }}
            transition={{ duration: 1.5, ease: 'easeOut' }}
          />
          
          <motion.circle 
            cx="400" 
            cy="4" 
            r="3" 
            fill="#3ddbd9" 
            filter="url(#lineGlowCompact)"
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: animated ? 1 : 0, opacity: animated ? 1 : 0 }}
            transition={{ duration: 0.5, ease: 'easeOut', delay: 1.4 }}
          />
        </svg>
        
        <motion.div 
          initial={{ opacity: 0, y: 5 }}
          animate={{ opacity: animated ? 1 : 0, y: animated ? 0 : 5 }}
          transition={{ duration: 0.5, ease: 'easeOut', delay: 1.6 }}
          className="absolute right-0 bottom-0 flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-green-500/10 border border-green-500/20"
        >
          <div className="w-1.5 h-1.5 rounded-full bg-green-400 shadow-lg shadow-green-400/50"></div>
          <span className="text-[10px] text-green-400 font-medium">Good signal</span>
        </motion.div>
      </div>
    </motion.div>
  )
}

