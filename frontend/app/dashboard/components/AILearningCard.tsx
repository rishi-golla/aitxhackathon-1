'use client'

import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'

const AlertIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
    <line x1="12" y1="9" x2="12" y2="13" />
    <line x1="12" y1="17" x2="12.01" y2="17" />
  </svg>
)

const DollarIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="12" y1="1" x2="12" y2="23" />
    <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
  </svg>
)

export default function AILearningCard() {
  const [totalCost, setTotalCost] = useState(0)
  const [violationCount, setViolationCount] = useState(0)
  const [isAnimating, setIsAnimating] = useState(false)
  
  // Base cost that grows over time
  const TARGET_COST = 16131
  const COST_PER_VIOLATION = 2500
  
  // Animate counter on mount
  useEffect(() => {
    const duration = 2000 // 2 seconds
    const steps = 60
    const increment = TARGET_COST / steps
    let current = 0
    
    const timer = setInterval(() => {
      current += increment
      if (current >= TARGET_COST) {
        setTotalCost(TARGET_COST)
        clearInterval(timer)
      } else {
        setTotalCost(Math.floor(current))
      }
    }, duration / steps)
    
    return () => clearInterval(timer)
  }, [])
  
  // Simulate new violations periodically (for demo)
  useEffect(() => {
    const violationTimer = setInterval(() => {
      setIsAnimating(true)
      setViolationCount(prev => prev + 1)
      setTotalCost(prev => prev + COST_PER_VIOLATION)
      
      setTimeout(() => setIsAnimating(false), 600)
    }, 15000) // New violation every 15 seconds
    
    return () => clearInterval(violationTimer)
  }, [])
  
  return (
    <div className="w-full rounded-2xl bg-card shadow-card border border-white/5 p-3 hover:shadow-card-hover hover:border-red-500/30 transition-all duration-300 animate-slide-up relative overflow-hidden" style={{ animationDelay: '200ms' }}>
      {/* Animated background pulse on new violation */}
      {isAnimating && (
        <motion.div
          initial={{ opacity: 0.3, scale: 0.8 }}
          animate={{ opacity: 0, scale: 1.5 }}
          transition={{ duration: 0.6 }}
          className="absolute inset-0 bg-red-500/20 rounded-2xl"
        />
      )}
      
      {/* Header */}
      <div className="flex items-start justify-between mb-2 relative z-10">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-red-500/10 flex items-center justify-center">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
              <line x1="12" y1="9" x2="12" y2="13" />
              <line x1="12" y1="17" x2="12.01" y2="17" />
            </svg>
          </div>
          <div>
            <h3 className="text-white text-[11px] font-semibold">Cost of Inaction</h3>
            <p className="text-white/50 text-[9px]">OSHA Violations</p>
          </div>
        </div>
      </div>
      
      {/* Main Counter */}
      <div className="relative z-10 mb-3">
        <div className="flex items-baseline gap-2">
          <motion.div
            key={totalCost}
            initial={{ scale: 1 }}
            animate={{ scale: isAnimating ? [1, 1.1, 1] : 1 }}
            transition={{ duration: 0.3 }}
            className="flex items-center"
          >
            <span className="text-red-400 text-xl font-bold tabular-nums">
              +${totalCost.toLocaleString()}
            </span>
          </motion.div>
          {isAnimating && (
            <motion.span
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: [0, 1, 0], y: -20 }}
              transition={{ duration: 0.6 }}
              className="text-red-400 text-xs font-semibold"
            >
              +${COST_PER_VIOLATION.toLocaleString()}
            </motion.span>
          )}
        </div>
        <p className="text-white/40 text-[9px] mt-0.5">
          Potential fines & liability costs
        </p>
      </div>
      
      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-2 relative z-10 mb-2">
        <div className="bg-white/5 rounded-lg p-1.5">
          <div className="text-white/50 text-[8px] mb-0.5">Violations</div>
          <motion.div
            key={violationCount}
            initial={{ scale: 1 }}
            animate={{ scale: isAnimating ? [1, 1.2, 1] : 1 }}
            className="text-red-400 text-base font-bold tabular-nums"
          >
            {violationCount + 7}
          </motion.div>
        </div>
        
        <div className="bg-white/5 rounded-lg p-1.5">
          <div className="text-white/50 text-[8px] mb-0.5">Avg Fine</div>
          <div className="text-amber-400 text-base font-bold tabular-nums">
            ${COST_PER_VIOLATION.toLocaleString()}
          </div>
        </div>
      </div>
      
      {/* Warning Footer */}
      <div className="pt-2 border-t border-white/5 relative z-10">
        <div className="flex items-center gap-1.5">
          <div className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />
          <p className="text-white/60 text-[9px]">
            Real-time violation monitoring active
          </p>
        </div>
      </div>
    </div>
  )
}
