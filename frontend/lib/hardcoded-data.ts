// Hardcoded data - shared across all instances
// This file contains the default layout that everyone sees

export interface Camera {
  id: string
  label: string
  streamUrl: string
  floor: number
  position: string // JSON string: [x, y, z]
  rotation: string // JSON string: [x, y, z]
  status: 'normal' | 'warning' | 'incident' | 'inactive'
  active: boolean
}

export interface WallSegment {
  id: string
  floor: number
  start: string // JSON string: [x, y]
  end: string   // JSON string: [x, y]
}

export interface FloorPlan {
  referenceImage: string | null
}

// Pre-placed cameras - Positioned strategically for warehouse layout
// The 3D scene normalizes the floor plan to roughly -7 to +7 on X axis and -3 to +3 on Z axis
// Cameras are placed at key monitoring points
export const HARDCODED_CAMERAS: Camera[] = [
  {
    id: 'cam-entrance',
    label: 'Main Entrance',
    streamUrl: 'rtsp://demo.com/stream1',
    floor: 1,
    position: JSON.stringify([-5, 0, 2.5]), // Top left - entrance area
    rotation: JSON.stringify([0, 0, 0]),
    status: 'normal',
    active: true,
  },
  {
    id: 'cam-warehouse-north',
    label: 'Warehouse North',
    streamUrl: 'rtsp://demo.com/stream2',
    floor: 1,
    position: JSON.stringify([0, 0, 2.5]), // Top center - main warehouse area
    rotation: JSON.stringify([0, 0, 0]),
    status: 'normal',
    active: true,
  },
  {
    id: 'cam-warehouse-south',
    label: 'Warehouse South',
    streamUrl: 'rtsp://demo.com/stream3',
    floor: 1,
    position: JSON.stringify([0, 0, -2.5]), // Bottom center - warehouse floor
    rotation: JSON.stringify([0, 0, 0]),
    status: 'normal',
    active: true,
  },
  {
    id: 'cam-loading-dock',
    label: 'Loading Dock',
    streamUrl: 'rtsp://demo.com/stream4',
    floor: 1,
    position: JSON.stringify([5, 0, -2.5]), // Bottom right - loading area
    rotation: JSON.stringify([0, 0, 0]),
    status: 'warning',
    active: true,
  },
]

// Default walls (empty - will be auto-traced from floor plan)
export const HARDCODED_WALLS: WallSegment[] = []

// Your uploaded floor plan
// Using the download.jpg file from the root directory
export const HARDCODED_FLOORPLAN: FloorPlan = {
  referenceImage: '/download.jpg', // Path to your floor plan image
}

// In-memory storage that persists during app runtime
let runtimeCameras: Camera[] = [...HARDCODED_CAMERAS]
let runtimeWalls: WallSegment[] = [...HARDCODED_WALLS]
let runtimeFloorPlan: FloorPlan = { ...HARDCODED_FLOORPLAN }

export function getCameras(): Camera[] {
  return runtimeCameras
}

export function setCameras(cameras: Camera[]) {
  runtimeCameras = cameras
}

export function getWalls(): WallSegment[] {
  return runtimeWalls
}

export function setWalls(walls: WallSegment[]) {
  runtimeWalls = walls
}

export function getFloorPlan(): FloorPlan {
  return runtimeFloorPlan
}

export function setFloorPlan(floorPlan: FloorPlan) {
  runtimeFloorPlan = floorPlan
}

export function resetToDefaults() {
  runtimeCameras = [...HARDCODED_CAMERAS]
  runtimeWalls = [...HARDCODED_WALLS]
  runtimeFloorPlan = { ...HARDCODED_FLOORPLAN }
}
