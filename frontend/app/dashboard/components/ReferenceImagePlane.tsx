'use client'

import { useTexture } from '@react-three/drei'
import * as THREE from 'three'

interface ReferenceImagePlaneProps {
  imageUrl: string | null
  visible: boolean
  onFloorClick?: (position: [number, number, number]) => void
}

function ImagePlane({ imageUrl, onFloorClick }: { imageUrl: string; onFloorClick?: (position: [number, number, number]) => void }) {
  const texture = useTexture(imageUrl)
  texture.minFilter = THREE.LinearFilter
  texture.magFilter = THREE.LinearFilter

  const handleClick = (event: any) => {
    if (onFloorClick) {
      event.stopPropagation()
      const point = event.point
      onFloorClick([point.x, 0, point.z])
    }
  }

  return (
    <mesh
      rotation={[-Math.PI / 2, 0, 0]}
      position={[0, 0, 0]}
      onClick={handleClick}
    >
      <planeGeometry args={[14, 14]} />
      <meshBasicMaterial
        map={texture}
        transparent
        opacity={0.2}
        depthWrite={false}
      />
    </mesh>
  )
}

export default function ReferenceImagePlane({ imageUrl, visible, onFloorClick }: ReferenceImagePlaneProps) {
  if (!imageUrl || !visible) return null

  return <ImagePlane imageUrl={imageUrl} onFloorClick={onFloorClick} />
}

