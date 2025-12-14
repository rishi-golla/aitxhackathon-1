// Hardcoded data for cameras, walls, and floor plan
// This is the default data that loads for everyone when they open the app

export interface Camera {
  id: string
  label: string
  streamUrl: string
  floor: number
  position: string // JSON stringified [x, y, z]
  rotation: string // JSON stringified [x, y, z]
  status: 'normal' | 'warning' | 'incident' | 'inactive'
  active: boolean
}

export interface WallSegment {
  id: string
  floor: number
  start: string // JSON stringified [x, z]
  end: string // JSON stringified [x, z]
}

export interface FloorPlan {
  referenceImage: string | null
}

// Default hardcoded cameras (7 cameras, one red/incident)
export const HARDCODED_CAMERAS: Camera[] = [
  {
    id: 'cam-entrance',
    label: 'Main Entrance',
    streamUrl: 'rtsp://demo.com/stream1',
    floor: 1,
    position: JSON.stringify([-4.5, 0, 1.8]), // Top left (within layout)
    rotation: JSON.stringify([0, 0, 0]),
    status: 'normal',
    active: true,
  },
  {
    id: 'cam-warehouse-north',
    label: 'Warehouse North',
    streamUrl: 'rtsp://demo.com/stream2',
    floor: 1,
    position: JSON.stringify([-2, 0, 1.8]), // Top center-left (within layout)
    rotation: JSON.stringify([0, 0, 0]),
    status: 'normal',
    active: true,
  },
  {
    id: 'cam-warehouse-center',
    label: 'Warehouse Center',
    streamUrl: 'rtsp://demo.com/stream3',
    floor: 1,
    position: JSON.stringify([0, 0, 1.5]), // Top center (within layout)
    rotation: JSON.stringify([0, 0, 0]),
    status: 'normal',
    active: true,
  },
  {
    id: 'cam-warehouse-east',
    label: 'Warehouse East',
    streamUrl: 'rtsp://demo.com/stream4',
    floor: 1,
    position: JSON.stringify([4, 0, 1.8]), // Top right (within layout)
    rotation: JSON.stringify([0, 0, 0]),
    status: 'normal',
    active: true,
  },
  {
    id: 'cam-warehouse-south',
    label: 'Warehouse South',
    streamUrl: 'rtsp://demo.com/stream5',
    floor: 1,
    position: JSON.stringify([-1.5, 0, -1.2]), // Bottom left (within layout)
    rotation: JSON.stringify([0, 0, 0]),
    status: 'normal',
    active: true,
  },
  {
    id: 'cam-loading-dock',
    label: 'Loading Dock',
    streamUrl: 'rtsp://demo.com/stream6',
    floor: 1,
    position: JSON.stringify([2, 0, -1.8]), // Bottom center-right (within layout)
    rotation: JSON.stringify([0, 0, 0]),
    status: 'warning',
    active: true,
  },
  {
    id: 'cam-hazard-zone',
    label: 'Hazard Zone - ALERT',
    streamUrl: 'rtsp://demo.com/stream7',
    floor: 1,
    position: JSON.stringify([4.5, 0, -1.8]), // Bottom right (within layout)
    rotation: JSON.stringify([0, 0, 0]),
    status: 'incident', // RED camera
    active: true,
  },
]

// Default hardcoded walls (empty by default, will auto-trace from image)
export const HARDCODED_WALLS: WallSegment[] = []

// Default hardcoded floor plan
export const HARDCODED_FLOORPLAN: FloorPlan = {
  referenceImage: '/download.jpg', // Path to your floor plan image
}

// In-memory storage (can be updated at runtime)
let currentCameras: Camera[] = [...HARDCODED_CAMERAS]
let currentWalls: WallSegment[] = [...HARDCODED_WALLS]
let currentFloorPlan: FloorPlan = { ...HARDCODED_FLOORPLAN }

// Getter functions
export function getCameras(): Camera[] {
  return currentCameras.length > 0 ? currentCameras : HARDCODED_CAMERAS
}

export function getWalls(): WallSegment[] {
  return currentWalls.length > 0 ? currentWalls : HARDCODED_WALLS
}

export function getFloorPlan(): FloorPlan {
  return currentFloorPlan.referenceImage ? currentFloorPlan : HARDCODED_FLOORPLAN
}

// Setter functions
export function setCameras(cameras: Camera[]): void {
  currentCameras = cameras.length > 0 ? cameras : HARDCODED_CAMERAS
}

export function setWalls(walls: WallSegment[]): void {
  currentWalls = walls
}

export function setFloorPlan(floorPlan: FloorPlan): void {
  currentFloorPlan = floorPlan
}

// Reset to defaults
export function resetToDefaults(): void {
  currentCameras = [...HARDCODED_CAMERAS]
  currentWalls = [...HARDCODED_WALLS]
  currentFloorPlan = { ...HARDCODED_FLOORPLAN }
}

