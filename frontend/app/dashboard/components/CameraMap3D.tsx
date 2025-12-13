'use client'

import { Suspense } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { useRef } from 'react'
import * as THREE from 'three'

function Floor() {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]} receiveShadow>
      <planeGeometry args={[10, 10]} />
      <meshStandardMaterial 
        color="#FFFFFF" 
        metalness={0.1}
        roughness={0.9}
      />
    </mesh>
  )
}

function Wall({ position, rotation, size }: { position: [number, number, number], rotation?: [number, number, number], size: [number, number] }) {
  return (
    <mesh position={position} rotation={rotation || [0, 0, 0]} castShadow receiveShadow>
      <boxGeometry args={size} />
      <meshStandardMaterial 
        color="#E5E7EB" 
        metalness={0.1}
        roughness={0.8}
      />
    </mesh>
  )
}

function CameraNode({ position, active = false }: { position: [number, number, number], active?: boolean }) {
  const meshRef = useRef<THREE.Mesh>(null)
  
  return (
    <group position={position}>
      <mesh ref={meshRef} castShadow>
        <sphereGeometry args={[0.06, 32, 32]} />
        <meshStandardMaterial 
          emissive="#28e7d3" 
          emissiveIntensity={active ? 3 : 2}
          color="#28e7d3"
          metalness={0.5}
          roughness={0.3}
        />
      </mesh>
      {active && (
        <mesh>
          <ringGeometry args={[0.08, 0.12, 32]} />
          <meshBasicMaterial 
            color="#28e7d3" 
            transparent 
            opacity={0.5}
            side={THREE.DoubleSide}
          />
        </mesh>
      )}
    </group>
  )
}

function Cameras() {
  // Camera positions in 3D space
  const cameraPositions: [number, number, number][] = [
    [-3, 0.1, -3],
    [-1, 0.1, -3],
    [1, 0.1, -3],
    [3, 0.1, -3],
    [-3, 0.1, 0],
    [3, 0.1, 0],
    [-3, 0.1, 3],
    [-1, 0.1, 3],
    [1, 0.1, 3],
    [3, 0.1, 3],
    [0, 0.1, -1],
    [0, 0.1, 1],
  ]
  
  return (
    <>
      {cameraPositions.map((pos, index) => (
        <CameraNode key={index} position={pos} active={index === 6} />
      ))}
    </>
  )
}

function Scene() {
  return (
    <>
      {/* Lighting - Brighter for white background */}
      <ambientLight intensity={0.8} />
      <pointLight position={[2, 5, 2]} intensity={0.8} color="#FFFFFF" />
      <pointLight position={[-2, 5, -2]} intensity={0.6} color="#FFFFFF" />
      <directionalLight position={[5, 5, 5]} intensity={0.5} castShadow />
      
      {/* Scene objects */}
      <Floor />
      
      {/* Building structure - walls */}
      <Wall position={[-4, 0.5, 0]} size={[0.2, 1, 8]} />
      <Wall position={[4, 0.5, 0]} size={[0.2, 1, 8]} />
      <Wall position={[0, 0.5, -4]} size={[8, 1, 0.2]} />
      <Wall position={[0, 0.5, 4]} size={[8, 1, 0.2]} />
      
      {/* Internal walls */}
      <Wall position={[0, 0.5, -1.5]} size={[6, 1, 0.15]} />
      <Wall position={[-2, 0.5, 1.5]} size={[2, 1, 0.15]} />
      <Wall position={[2, 0.5, 1.5]} size={[2, 1, 0.15]} />
      
      {/* Camera nodes */}
      <Cameras />
      
      {/* Camera preview area highlight */}
      <mesh position={[-1, 0.05, 3]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[1.5, 1]} />
        <meshStandardMaterial 
          color="#7066FF" 
          emissive="#7066FF"
          emissiveIntensity={0.2}
          transparent
          opacity={0.15}
        />
      </mesh>
      
      {/* Controls */}
      <OrbitControls 
        enablePan={false} 
        maxPolarAngle={Math.PI / 2.2}
        minDistance={5}
        maxDistance={12}
        enableZoom={true}
        autoRotate={false}
      />
    </>
  )
}

export default function CameraMap3D() {
  return (
    <div className="w-full h-full">
      <Suspense fallback={
        <div className="w-full h-full flex items-center justify-center bg-white">
          <div className="text-gray-400 text-sm">Loading 3D map...</div>
        </div>
      }>
        <Canvas
          shadows
          gl={{ antialias: true, alpha: true }}
          camera={{ position: [0, 4, 6], fov: 40 }}
        >
          <Scene />
        </Canvas>
      </Suspense>
    </div>
  )
}
