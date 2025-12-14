// Hardcoded data - shared across all instances
// Edit this file to change the default layout for everyone

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

// Default hardcoded cameras
export const HARDCODED_CAMERAS: Camera[] = [
  // Example cameras - edit these or add more
  // {
  //   id: 'cam-1',
  //   label: 'Entrance Camera',
  //   streamUrl: 'https://example.com/stream1',
  //   floor: 1,
  //   position: JSON.stringify([2, 0, 3]),
  //   rotation: JSON.stringify([0, 0, 0]),
  //   status: 'normal',
  //   active: true,
  // },
]

// Default hardcoded walls
export const HARDCODED_WALLS: WallSegment[] = [
  // Example walls - edit these or add more
  // {
  //   id: 'wall-1',
  //   floor: 1,
  //   start: JSON.stringify([0, 0]),
  //   end: JSON.stringify([10, 0]),
  // },
]

// Default hardcoded floor plan
export const HARDCODED_FLOORPLAN: FloorPlan = {
  referenceImage: null, // Set to a base64 image or URL if you want a default floor plan
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

