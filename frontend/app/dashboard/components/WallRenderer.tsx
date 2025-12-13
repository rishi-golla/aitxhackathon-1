'use client'

import { Line } from '@react-three/drei'
import { WallSegment } from '../types/floorplan'
import { useThree } from '@react-three/fiber'

interface WallRendererProps {
  walls: WallSegment[]
  currentFloor: number
}

export default function WallRenderer({ walls, currentFloor }: WallRendererProps) {
  const wallHeight = 0.4
  const wallThickness = 0.04
  const elevation = 0.02
  const { camera } = useThree()

  const visibleWalls = walls.filter(w => w.floor === currentFloor)

  return (
    <>
      {visibleWalls.map((wall) => {
        const [x1, z1] = wall.start
        const [x2, z2] = wall.end
        
        const length = Math.sqrt((x2 - x1) ** 2 + (z2 - z1) ** 2)
        if (length < 0.01) return null
        
        const angle = Math.atan2(z2 - z1, x2 - x1)
        const midX = (x1 + x2) / 2
        const midZ = (z1 + z2) / 2

        // Calculate distance from camera for depth-based fading
        const distanceFromCamera = Math.sqrt(
          Math.pow(camera.position.x - midX, 2) +
          Math.pow(camera.position.z - midZ, 2)
        )
        const maxDistance = 18
        const fadeStart = 12
        const fadeFactor = distanceFromCamera > fadeStart 
          ? Math.max(0.3, 1 - (distanceFromCamera - fadeStart) / (maxDistance - fadeStart))
          : 1

        return (
          <group key={wall.id}>
            {/* Wall solid - muted to let cameras stand out */}
            <mesh position={[midX, wallHeight / 2 + elevation, midZ]} rotation={[0, angle, 0]}>
              <boxGeometry args={[length, wallHeight, wallThickness]} />
              <meshStandardMaterial
                color="#3a4350"
                emissive="#2a3340"
                emissiveIntensity={0.15}
                metalness={0}
                roughness={0.95}
                transparent
                opacity={0.75 * fadeFactor}
              />
            </mesh>
            
            {/* Top edge line - subtle */}
            <Line
              points={[
                [x1, wallHeight + elevation, z1],
                [x2, wallHeight + elevation, z2],
              ]}
              color="#3ddbd9"
              lineWidth={1.5}
              transparent
              opacity={0.4 * fadeFactor}
            />
            
            {/* Bottom edge line */}
            <Line
              points={[
                [x1, elevation, z1],
                [x2, elevation, z2],
              ]}
              color="#28e7d3"
              lineWidth={1}
              transparent
              opacity={0.3 * fadeFactor}
            />
          </group>
        )
      })}
    </>
  )
}

