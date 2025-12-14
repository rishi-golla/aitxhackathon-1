import { NextResponse } from 'next/server'
import { getCameras, getWalls } from '@/lib/hardcoded-data'

export async function GET() {
  const cameras = getCameras()
  const walls = getWalls()

  return NextResponse.json({
    status: 'ok',
    message: 'API is working',
    database: 'hardcoded',
    databaseType: 'Hardcoded',
    cameraCount: cameras.length,
    wallCount: walls.length,
    note: 'Using hardcoded data from frontend/lib/hardcoded-data.ts. No database required!',
  })
}
