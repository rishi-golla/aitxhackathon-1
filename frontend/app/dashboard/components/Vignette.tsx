'use client'

import { useThree } from '@react-three/fiber'
import { useEffect, useRef } from 'react'
import * as THREE from 'three'

export default function Vignette() {
  const { camera } = useThree()
  const vignetteRef = useRef<THREE.Mesh>(null)

  useEffect(() => {
    if (vignetteRef.current) {
      // Position vignette in front of camera
      vignetteRef.current.position.copy(camera.position)
      vignetteRef.current.position.z -= 5
      vignetteRef.current.lookAt(camera.position)
    }
  }, [camera])

  return (
    <mesh ref={vignetteRef} position={[0, 0, -5]}>
      <planeGeometry args={[30, 30]} />
      <meshBasicMaterial
        transparent
        opacity={0.4}
        depthWrite={false}
        side={THREE.DoubleSide}
      >
        <primitive 
          attach="map" 
          object={(() => {
            const canvas = document.createElement('canvas')
            canvas.width = 512
            canvas.height = 512
            const ctx = canvas.getContext('2d')
            if (ctx) {
              const gradient = ctx.createRadialGradient(256, 256, 0, 256, 256, 256)
              gradient.addColorStop(0, 'rgba(0,0,0,0)')
              gradient.addColorStop(0.6, 'rgba(0,0,0,0)')
              gradient.addColorStop(1, 'rgba(2,6,23,0.6)')
              ctx.fillStyle = gradient
              ctx.fillRect(0, 0, 512, 512)
            }
            const texture = new THREE.CanvasTexture(canvas)
            return texture
          })()} 
        />
      </meshBasicMaterial>
    </mesh>
  )
}

