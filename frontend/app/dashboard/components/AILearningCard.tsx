'use client'

import { useEffect, useState } from 'react'

const ArrowIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M6 4L10 8L6 12" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
)

interface ProgressBarProps {
  label: string
  value: number
  max?: number
  delay?: number
}

const ProgressBar = ({ label, value, max = 5000, delay = 0 }: ProgressBarProps) => {
  const [animated, setAnimated] = useState(false)
  const percentage = (value / max) * 100
  
  useEffect(() => {
    const timer = setTimeout(() => setAnimated(true), delay)
    return () => clearTimeout(timer)
  }, [delay])
  
  return (
    <div className="flex items-center gap-2">
      <span className="text-[9px] text-white/60 w-16 shrink-0">{label}</span>
      <div className="h-1.5 rounded-full bg-white/5 overflow-hidden flex-1">
        <div 
          className="h-full rounded-full transition-all duration-1000 ease-out"
          style={{ 
            width: animated ? `${percentage}%` : '0%',
            background: 'linear-gradient(90deg, #28e7d3 0%, #00F5CC 100%)',
            boxShadow: '0 0 6px rgba(40, 231, 211, 0.3)'
          }}
        />
      </div>
      <span className="text-[9px] text-white/80 w-10 text-right font-medium tabular-nums">
        {value.toLocaleString()}
      </span>
    </div>
  )
}

export default function AILearningCard() {
  const metrics = [
    { label: 'Scanned', value: 4294 },
    { label: 'Recognised', value: 3980 },
    { label: 'Known', value: 3105 },
    { label: 'Learned', value: 790 },
  ]
  
  return (
    <div className="w-full rounded-2xl bg-card shadow-card border border-white/5 p-3 hover:shadow-card-hover hover:border-cyan/20 transition-all duration-300 animate-slide-up" style={{ animationDelay: '200ms' }}>
      {/* Header */}
      <div className="flex items-start justify-between mb-2">
        <h3 className="text-white/80 text-[10px] font-medium uppercase tracking-wide">AI learning</h3>
        <button className="text-white/30 hover:text-cyan transition-colors">
          <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M6 4L10 8L6 12" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
      </div>
      
      {/* Progress Bars */}
      <div className="space-y-2">
        {metrics.map((metric, index) => (
          <ProgressBar 
            key={metric.label} 
            label={metric.label} 
            value={metric.value}
            delay={index * 150}
          />
        ))}
      </div>
    </div>
  )
}
