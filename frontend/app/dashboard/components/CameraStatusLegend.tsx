'use client'

export default function CameraStatusLegend() {
  const statuses = [
    { color: '#3ddbd9', label: 'Normal', bgClass: 'bg-cyan-400' },
    { color: '#f59e0b', label: 'Warning', bgClass: 'bg-amber-400' },
    { color: '#ef4444', label: 'Incident', bgClass: 'bg-red-400' },
    { color: '#6b7280', label: 'Inactive', bgClass: 'bg-gray-400' },
  ]

  return (
    <div className="absolute bottom-3 left-3 px-3 py-2 bg-zinc-900/90 border border-white/10 rounded-lg backdrop-blur-xl shadow-lg">
      <div className="flex items-center gap-3">
        {statuses.map((status) => (
          <div key={status.label} className="flex items-center gap-1.5">
            <div 
              className={`w-2 h-2 rounded-full ${status.bgClass}`}
              style={{ 
                boxShadow: `0 0 8px ${status.color}`,
              }}
            />
            <span className="text-[10px] text-zinc-400 font-medium">{status.label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

