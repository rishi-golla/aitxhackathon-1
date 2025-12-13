// Floor plan data model

export type WallSegment = {
  id: string
  start: [number, number]
  end: [number, number]
  floor: number
}

export type CameraNode = {
  id: string
  position: [number, number, number]
  rotation: [number, number, number]
  floor: number
  label: string
  streamUrl: string
  active: boolean
}

export type FloorPlanVectors = {
  walls: WallSegment[]
  cameras: CameraNode[]
  referenceImage: string | null
}

export type InteractionMode = 'view' | 'place-cameras'

