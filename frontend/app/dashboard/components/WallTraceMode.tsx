'use client'

import { useState, useEffect } from 'react'
import { Line } from '@react-three/drei'
import { WallSegment } from '../types/floorplan'

interface WallTraceModeProps {
  active: boolean
  onWallsComplete: (walls: WallSegment[]) => void
  currentFloor: number
}

export default function WallTraceMode({ active, onWallsComplete, currentFloor }: WallTraceModeProps) {
  const [points, setPoints] = useState<[number, number][]>([])
  const [previewPoint, setPreviewPoint] = useState<[number, number] | null>(null)

  useEffect(() => {
    if (!active) {
      setPoints([])
      setPreviewPoint(null)
    }
  }, [active])

  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (!active) return
      
      if (e.key === 'Enter' && points.length >= 2) {
        completeTracing()
      } else if (e.key === 'Escape') {
        setPoints([])
        setPreviewPoint(null)
      } else if (e.key === 'z' && e.ctrlKey && points.length > 0) {
        setPoints(prev => prev.slice(0, -1))
      }
    }

    window.addEventListener('keydown', handleKeyPress)
    return () => window.removeEventListener('keydown', handleKeyPress)
  }, [active, points])

  const completeTracing = () => {
    if (points.length < 2) return

    const walls: WallSegment[] = []
    for (let i = 0; i < points.length - 1; i++) {
      walls.push({
        id: `wall-${Date.now()}-${i}`,
        start: points[i],
        end: points[i + 1],
        floor: currentFloor,
      })
    }

    onWallsComplete(walls)
    setPoints([])
    setPreviewPoint(null)
  }

  const handleFloorClick = (position: [number, number, number]) => {
    if (!active) return

    const point2D: [number, number] = [position[0], position[2]]
    
    // Snap to nearby endpoint
    const snapThreshold = 0.3
    let snappedPoint = point2D
    
    if (points.length > 0) {
      const lastPoint = points[points.length - 1]
      const dist = Math.sqrt(
        Math.pow(point2D[0] - lastPoint[0], 2) + 
        Math.pow(point2D[1] - lastPoint[1], 2)
      )
      
      if (dist < snapThreshold) {
        snappedPoint = lastPoint
      }
    }

    setPoints(prev => [...prev, snappedPoint])
  }

  if (!active) return null

  return (
    <group>
      {/* Drawn segments */}
      {points.map((point, i) => {
        if (i === 0) return null
        const prevPoint = points[i - 1]
        return (
          <Line
            key={i}
            points={[
              [prevPoint[0], 0.02, prevPoint[1]],
              [point[0], 0.02, point[1]],
            ]}
            color="#3ddbd9"
            lineWidth={3}
          />
        )
      })}

      {/* Preview line */}
      {points.length > 0 && previewPoint && (
        <Line
          points={[
            [points[points.length - 1][0], 0.02, points[points.length - 1][1]],
            [previewPoint[0], 0.02, previewPoint[1]],
          ]}
          color="#3ddbd9"
          lineWidth={2}
          transparent
          opacity={0.5}
          dashed
          dashSize={0.2}
          gapSize={0.1}
        />
      )}

      {/* Point markers */}
      {points.map((point, i) => (
        <mesh key={`marker-${i}`} position={[point[0], 0.05, point[1]]}>
          <sphereGeometry args={[0.08, 16, 16]} />
          <meshStandardMaterial
            color="#3ddbd9"
            emissive="#3ddbd9"
            emissiveIntensity={2}
          />
        </mesh>
      ))}

      {/* Invisible floor plane for raycasting */}
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0, 0, 0]}
        onClick={(e) => {
          e.stopPropagation()
          handleFloorClick([e.point.x, 0, e.point.z])
        }}
        onPointerMove={(e) => {
          if (points.length > 0) {
            setPreviewPoint([e.point.x, e.point.z])
          }
        }}
      >
        <planeGeometry args={[14, 14]} />
        <meshBasicMaterial visible={false} />
      </mesh>
    </group>
  )
}

