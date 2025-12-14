import { NextRequest, NextResponse } from 'next/server'
import { 
  getCameras, 
  setCameras, 
  getWalls, 
  setWalls, 
  getFloorPlan, 
  setFloorPlan,
  Camera,
  WallSegment
} from '@/lib/hardcoded-data'

export async function GET(request: NextRequest) {
  try {
    const cameras = getCameras()
    const walls = getWalls()
    const floorPlan = getFloorPlan()

    console.log(`✅ Loaded hardcoded data: ${cameras.length} cameras, ${walls.length} walls`)

    return NextResponse.json({ 
      cameras, 
      walls, 
      floorPlan, 
      databaseType: 'Hardcoded' 
    })
  } catch (error) {
    console.error('Get user data error:', error)
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 })
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { cameras, walls, referenceImage } = body

    // Update in-memory data
    if (cameras && Array.isArray(cameras)) {
      setCameras(cameras as Camera[])
    }

    if (walls && Array.isArray(walls)) {
      setWalls(walls as WallSegment[])
    }

    if (referenceImage !== undefined) {
      setFloorPlan({ referenceImage })
    }

    console.log(`✅ Saved hardcoded data: ${cameras?.length || 0} cameras, ${walls?.length || 0} walls`)

    return NextResponse.json({ 
      status: 'ok', 
      databaseType: 'Hardcoded',
      note: 'Data saved in memory. Will reset when server restarts. Edit frontend/lib/hardcoded-data.ts to set permanent defaults.'
    })
  } catch (error) {
    console.error('Save user data error:', error)
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 })
  }
}
