'use client'

import { motion } from 'framer-motion'
import { useEffect, useState } from 'react'

const data = [
  { day: 'Mo', value: 12 },
  { day: 'Tu', value: 18 },
  { day: 'We', value: 14 },
  { day: 'Th', value: 20 },
  { day: 'Fr', value: 16 },
  { day: 'Sa', value: 28 },
  { day: 'Su', value: 22 },
]

export default function AccidentsCardCompact() {
  const [count, setCount] = useState(0)
  const [animated, setAnimated] = useState(false)
  const max = Math.max(...data.map(d => d.value))

  useEffect(() => {
    const duration = 1000
    const steps = 60
    const increment = 32 / steps
    let current = 0
    
    const timer = setInterval(() => {
      current += increment
      if (current >= 32) {
        setCount(32)
        clearInterval(timer)
      } else {
        setCount(Math.floor(current))
      }
    }, duration / steps)
    
    setTimeout(() => setAnimated(true), 300)
    
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
        <span className="text-xs tracking-wide text-zinc-400 uppercase">Violations</span>
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-zinc-500">
          <path d="M6 4L10 8L6 12" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </div>

      <div className="flex items-center gap-2 mb-3">
        <motion.span 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.4 }}
          className="text-3xl font-semibold text-white"
        >
          {count}
        </motion.span>
        <div className="flex items-center gap-1 rounded-full bg-red-500/10 px-1.5 py-0.5 text-xs font-medium text-red-400">
          <svg width="8" height="8" viewBox="0 0 10 10" fill="currentColor">
            <path d="M5 2L8 7H2L5 2Z"/>
          </svg>
          5%
        </div>
      </div>

      <div className="flex items-end justify-between gap-1.5 flex-1">
        {data.map((item, i) => {
          const height = (item.value / max) * 100
          const isHighlight = item.day === 'Sa' || item.day === 'Fr'

          return (
            <div key={item.day} className="flex flex-col items-center gap-1 flex-1">
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: animated ? `${height}%` : '0%', opacity: animated ? 1 : 0 }}
                transition={{ delay: i * 0.08, duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
                className="relative w-full rounded-t-md overflow-hidden"
                style={{
                  minHeight: '3px',
                  background: isHighlight 
                    ? 'linear-gradient(180deg, #3ddbd9 0%, #00F5CC 100%)'
                    : 'linear-gradient(180deg, rgba(61,219,217,0.7) 0%, rgba(0,245,204,0.7) 100%)',
                  boxShadow: isHighlight 
                    ? '0 0 12px rgba(61,219,217,0.4)'
                    : '0 0 6px rgba(61,219,217,0.2)',
                }}
              />
              <motion.span 
                initial={{ opacity: 0 }}
                animate={{ opacity: animated ? 1 : 0 }}
                transition={{ delay: i * 0.08 + 0.3, duration: 0.3 }}
                className="text-[9px] font-medium text-zinc-500"
              >
                {item.day}
              </motion.span>
            </div>
          )
        })}
      </div>
    </motion.div>
  )
}

