'use client'

import { useRef, useState } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { CameraNode } from '../types/floorplan'

interface CameraNodeRendererProps {
  cameras: CameraNode[]
  currentFloor: number
  selectedCamera: string | null
  onCameraClick: (id: string) => void
}

// Semantic color states
const getCameraState = (camera: CameraNode) => {
  // Demo: Randomly assign states for demonstration
  // In production, this would come from camera.status or real-time data
  const hash = camera.id.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0)
  const rand = hash % 100
  
  if (!camera.active) {
    return { color: '#6b7280', label: 'inactive', intensity: 1.5 }
  } else if (rand < 10) {
    return { color: '#ef4444', label: 'incident', intensity: 4 }
  } else if (rand < 25) {
    return { color: '#f59e0b', label: 'warning', intensity: 3 }
  } else {
    return { color: '#3ddbd9', label: 'normal', intensity: 3 }
  }
}

function CameraDot({ camera, isSelected, onClick }: { camera: CameraNode; isSelected: boolean; onClick: () => void }) {
  const meshRef = useRef<THREE.Mesh>(null)
  const ringRef = useRef<THREE.Mesh>(null)
  const glowRef = useRef<THREE.PointLight>(null)
  const [hovered, setHovered] = useState(false)
  const { camera: sceneCamera } = useThree()
  
  const state = getCameraState(camera)

  useFrame(({ clock }) => {
    if (meshRef.current) {
      // Gentle floating animation
      meshRef.current.position.y = camera.position[1] + Math.sin(clock.elapsedTime * 2) * 0.08
      
      const targetScale = hovered || isSelected ? 1.8 : 1
      meshRef.current.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.1)
      
      const material = meshRef.current.material as THREE.MeshStandardMaterial
      const baseIntensity = state.intensity * 1.5
      // Add subtle pulse to all cameras for visibility
      const pulseAmount = Math.sin(clock.elapsedTime * 1.5) * 0.3
      const targetEmissiveIntensity = hovered || isSelected 
        ? baseIntensity * 1.5 
        : baseIntensity + pulseAmount
      material.emissiveIntensity = THREE.MathUtils.lerp(material.emissiveIntensity, targetEmissiveIntensity, 0.1)
    }

    if (ringRef.current && (hovered || isSelected)) {
      ringRef.current.rotation.z = clock.elapsedTime * 0.5
      ringRef.current.scale.setScalar(1 + Math.sin(clock.elapsedTime * 3) * 0.15)
    }

    // Depth-based fading (less aggressive)
    if (meshRef.current) {
      const distanceFromCamera = Math.sqrt(
        Math.pow(sceneCamera.position.x - camera.position[0], 2) +
        Math.pow(sceneCamera.position.z - camera.position[2], 2)
      )
      const maxDistance = 18
      const fadeStart = 14
      const fadeFactor = distanceFromCamera > fadeStart 
        ? Math.max(0.6, 1 - (distanceFromCamera - fadeStart) / (maxDistance - fadeStart))
        : 1
      
      const material = meshRef.current.material as THREE.MeshStandardMaterial
      material.opacity = fadeFactor
    }

    // Strong pulsing for incidents
    if (glowRef.current) {
      if (state.label === 'incident') {
        glowRef.current.intensity = 1.5 + Math.sin(clock.elapsedTime * 4) * 0.8
      } else {
        // Subtle pulse for all cameras
        glowRef.current.intensity = (state.label === 'incident' ? 1.5 : 1.0) + Math.sin(clock.elapsedTime * 1.5) * 0.2
      }
    }
  })

  return (
    <group 
      position={camera.position}
      onClick={onClick}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
    >
      {/* Enhanced glow light - much brighter */}
      <pointLight 
        ref={glowRef}
        position={[0, 0, 0]} 
        color={state.color} 
        intensity={state.label === 'incident' ? 1.5 : 1.0} 
        distance={3}
      />

      {/* Outer glow ring - always visible */}
      <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
        <ringGeometry args={[0.12, 0.18, 32]} />
        <meshBasicMaterial
          color={state.color}
          transparent
          opacity={0.3}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Main camera sphere - larger and brighter */}
      <mesh ref={meshRef}>
        <sphereGeometry args={[0.12, 32, 32]} />
        <meshStandardMaterial
          color={state.color}
          emissive={state.color}
          emissiveIntensity={state.intensity * 1.5}
          metalness={0.3}
          roughness={0.1}
          transparent
        />
      </mesh>

      {/* Inner bright core */}
      <mesh>
        <sphereGeometry args={[0.06, 16, 16]} />
        <meshBasicMaterial
          color="#ffffff"
          transparent
          opacity={0.8}
        />
      </mesh>

      {/* Animated pulsing ring for hover/select */}
      {(hovered || isSelected) && (
        <mesh ref={ringRef} rotation={[Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.20, 0.28, 32]} />
          <meshBasicMaterial
            color={state.color}
            transparent
            opacity={0.6}
            side={THREE.DoubleSide}
          />
        </mesh>
      )}

      {/* Vertical beam indicator - always visible, brighter */}
      <mesh position={[0, 0.6, 0]}>
        <cylinderGeometry args={[0.01, 0.01, 1.2, 8]} />
        <meshBasicMaterial 
          color={state.color} 
          transparent 
          opacity={state.label === 'warning' || state.label === 'incident' ? 0.6 : 0.4}
        />
      </mesh>

      {/* Top cap for beam */}
      <mesh position={[0, 1.2, 0]}>
        <sphereGeometry args={[0.04, 16, 16]} />
        <meshBasicMaterial
          color={state.color}
          transparent
          opacity={0.8}
        />
      </mesh>

      {/* Label background for better visibility */}
      {(hovered || isSelected) && (
        <group position={[0, 1.5, 0]}>
          <mesh>
            <planeGeometry args={[0.8, 0.2]} />
            <meshBasicMaterial
              color="#000000"
              transparent
              opacity={0.7}
            />
          </mesh>
        </group>
      )}
    </group>
  )
}

export default function CameraNodeRenderer({ cameras, currentFloor, selectedCamera, onCameraClick }: CameraNodeRendererProps) {
  const visibleCameras = cameras.filter(c => c.floor === currentFloor)

  return (
    <>
      {visibleCameras.map((camera) => (
        <CameraDot
          key={camera.id}
          camera={camera}
          isSelected={selectedCamera === camera.id}
          onClick={() => onCameraClick(camera.id)}
        />
      ))}
    </>
  )
}

