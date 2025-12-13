import { NextRequest, NextResponse } from 'next/server'
import { prisma } from '@/lib/prisma'
import { inMemoryDB } from '@/lib/db-fallback'
import jwt from 'jsonwebtoken'

const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key-change-in-production'

function getUserFromToken(request: NextRequest) {
  const token = request.headers.get('authorization')?.replace('Bearer ', '')
  
  if (!token) {
    return null
  }

  try {
    const decoded = jwt.verify(token, JWT_SECRET) as {
      userId: string
      email: string
    }
    return decoded
  } catch (error) {
    return null
  }
}

// GET user data (cameras, walls, floor plan)
export async function GET(request: NextRequest) {
  try {
    const decoded = getUserFromToken(request)

    if (!decoded) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      )
    }

    let data
    let dbType = 'In-Memory'

    try {
      // Try MongoDB first
      const [cameras, walls, floorPlan] = await Promise.all([
        prisma.camera.findMany({ where: { userId: decoded.userId } }),
        prisma.wallSegment.findMany({ where: { userId: decoded.userId } }),
        prisma.floorPlan.findFirst({ where: { userId: decoded.userId } }),
      ])

      data = { cameras, walls, floorPlan }
      dbType = 'MongoDB'
      console.log('✅ Data loaded from MongoDB')
    } catch (mongoError) {
      console.warn('⚠️ MongoDB unavailable, using in-memory storage')
      // Fallback to in-memory
      data = await inMemoryDB.getUserData(decoded.userId)
    }

    return NextResponse.json({ ...data, databaseType: dbType })
  } catch (error) {
    console.error('Get user data error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}

// POST - Save user data (cameras, walls, floor plan)
export async function POST(request: NextRequest) {
  try {
    const decoded = getUserFromToken(request)

    if (!decoded) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      )
    }

    const body = await request.json()
    const { cameras, walls, referenceImage } = body

    let dbType = 'In-Memory'

    try {
      // Try MongoDB first
      // Clear existing data
      await Promise.all([
        prisma.camera.deleteMany({ where: { userId: decoded.userId } }),
        prisma.wallSegment.deleteMany({ where: { userId: decoded.userId } }),
      ])

      // Save new cameras
      if (cameras && Array.isArray(cameras)) {
        await prisma.camera.createMany({
          data: cameras.map((camera: any) => ({
            userId: decoded.userId,
            label: camera.label,
            streamUrl: camera.streamUrl,
            floor: camera.floor,
            position: camera.position,
            rotation: camera.rotation,
            status: camera.status || 'normal',
            active: camera.active !== false,
          })),
        })
      }

      // Save new walls
      if (walls && Array.isArray(walls)) {
        await prisma.wallSegment.createMany({
          data: walls.map((wall: any) => ({
            userId: decoded.userId,
            floor: wall.floor,
            start: wall.start,
            end: wall.end,
          })),
        })
      }

      // Save floor plan
      const existingFloorPlan = await prisma.floorPlan.findFirst({
        where: { userId: decoded.userId },
      })

      if (existingFloorPlan) {
        await prisma.floorPlan.update({
          where: { id: existingFloorPlan.id },
          data: { referenceImage },
        })
      } else {
        await prisma.floorPlan.create({
          data: {
            userId: decoded.userId,
            referenceImage,
          },
        })
      }

      dbType = 'MongoDB'
      console.log('✅ Data saved to MongoDB')
    } catch (mongoError) {
      console.warn('⚠️ MongoDB unavailable, using in-memory storage')
      
      // Fallback to in-memory
      await inMemoryDB.deleteCamerasByUserId(decoded.userId)
      await inMemoryDB.deleteWallSegmentsByUserId(decoded.userId)

      if (cameras && Array.isArray(cameras)) {
        for (const camera of cameras) {
          await inMemoryDB.createCamera(decoded.userId, {
            label: camera.label,
            streamUrl: camera.streamUrl,
            floor: camera.floor,
            position: camera.position,
            rotation: camera.rotation,
            status: camera.status || 'normal',
            active: camera.active !== false,
          })
        }
      }

      if (walls && Array.isArray(walls)) {
        for (const wall of walls) {
          await inMemoryDB.createWallSegment(decoded.userId, {
            floor: wall.floor,
            start: wall.start,
            end: wall.end,
          })
        }
      }

      if (referenceImage !== undefined) {
        await inMemoryDB.updateFloorPlan(decoded.userId, referenceImage)
      }
    }

    return NextResponse.json({
      success: true,
      message: 'Data saved successfully',
      databaseType: dbType,
    })
  } catch (error) {
    console.error('Save user data error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}
