'use client'

import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import WallRenderer from './WallRenderer'
import CameraNodeRenderer from './CameraNodeRenderer'
import Vignette from './Vignette'
import { WallSegment, CameraNode, InteractionMode } from '../types/floorplan'

interface FactorySceneManualProps {
  referenceImage: string | null
  walls: WallSegment[]
  cameras: CameraNode[]
  currentFloor: number
  mode: InteractionMode
  selectedCamera: string | null
  onCameraClick: (id: string) => void
  onCameraPlace?: (position: [number, number, number]) => void
}

export default function FactorySceneManual({
  referenceImage,
  walls,
  cameras,
  currentFloor,
  mode,
  selectedCamera,
  onCameraClick,
  onCameraPlace,
}: FactorySceneManualProps) {
  
  const handleFloorClick = (position: [number, number, number]) => {
    if (mode === 'place-cameras' && onCameraPlace) {
      onCameraPlace(position)
    }
  }

  return (
    <Canvas
      shadows={false}
      camera={{ position: [12, 10, 12], fov: 40 }}
      gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }}
    >
      <color attach="background" args={['#1a1a1a']} />
      
      {/* Depth fog for spatial feel */}
      <fog attach="fog" args={['#141414', 8, 18]} />
      
      {/* Enhanced lighting with depth */}
      <ambientLight intensity={0.6} />
      <directionalLight position={[10, 12, 8]} intensity={0.6} />
      <pointLight position={[-8, 5, -8]} intensity={0.4} color="#3ddbd9" distance={18} />
      <pointLight position={[8, 5, 8]} intensity={0.4} color="#3ddbd9" distance={18} />
      <pointLight position={[0, 8, 0]} intensity={0.2} color="#ffffff" distance={20} />

      {/* Depth planes for spatial layering */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.02, 0]}>
        <planeGeometry args={[16, 16]} />
        <meshStandardMaterial color="#141414" metalness={0} roughness={1} transparent opacity={0.8} />
      </mesh>

      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]}>
        <planeGeometry args={[15, 15]} />
        <meshStandardMaterial color="#181818" metalness={0} roughness={1} transparent opacity={0.9} />
      </mesh>

      {/* Base floor */}
      <mesh 
        rotation={[-Math.PI / 2, 0, 0]} 
        position={[0, 0, 0]} 
        onClick={(e) => {
          if (mode === 'place-cameras') {
            e.stopPropagation()
            handleFloorClick([e.point.x, 0.15, e.point.z])
          }
        }}
      >
        <planeGeometry args={[14, 14]} />
        <meshStandardMaterial color="#1f1f1f" metalness={0} roughness={1} />
      </mesh>

      {/* Invisible larger clickable plane for camera placement */}
      {mode === 'place-cameras' && (
        <mesh 
          rotation={[-Math.PI / 2, 0, 0]} 
          position={[0, 0.001, 0]}
          onClick={(e) => {
            e.stopPropagation()
            handleFloorClick([e.point.x, 0.15, e.point.z])
          }}
        >
          <planeGeometry args={[20, 20]} />
          <meshBasicMaterial visible={false} />
        </mesh>
      )}

      {/* De-emphasized grid */}
      <gridHelper
        args={[14, 28, '#3ddbd9', '#2a2e35']}
        position={[0, 0.01, 0]}
        material-transparent
        material-opacity={0.06}
      />

      {/* Rendered walls */}
      <WallRenderer walls={walls} currentFloor={currentFloor} />

      {/* Camera nodes */}
      <CameraNodeRenderer
        cameras={cameras}
        currentFloor={currentFloor}
        selectedCamera={selectedCamera}
        onCameraClick={onCameraClick}
      />

      {/* Subtle vignette for depth */}
      <Vignette />

      {/* Camera controls */}
      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        enableDamping={true}
        dampingFactor={0.05}
        maxPolarAngle={Math.PI / 2.2}
        minPolarAngle={Math.PI / 6}
        minDistance={8}
        maxDistance={25}
        target={[0, 0, 0]}
        makeDefault
      />
    </Canvas>
  )
}

