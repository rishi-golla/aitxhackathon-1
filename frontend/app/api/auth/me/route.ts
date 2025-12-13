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

export async function GET(request: NextRequest) {
  try {
    const decoded = getUserFromToken(request)

    if (!decoded) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      )
    }

    let user
    let dbType = 'In-Memory'

    try {
      // Try MongoDB first
      user = await prisma.user.findUnique({
        where: { id: decoded.userId },
      })
      
      if (user) {
        dbType = 'MongoDB'
        console.log('✅ User profile loaded from MongoDB')
      }
    } catch (mongoError) {
      console.warn('⚠️ MongoDB unavailable, checking in-memory storage')
    }

    // Fallback to in-memory if not found in MongoDB
    if (!user) {
      user = await inMemoryDB.findUserById(decoded.userId)
    }

    if (!user) {
      return NextResponse.json(
        { error: 'User not found' },
        { status: 404 }
      )
    }

    const { password: _, ...userWithoutPassword } = user

    return NextResponse.json({ 
      user: userWithoutPassword,
      databaseType: dbType 
    })
  } catch (error) {
    console.error('Get user error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}

export async function PUT(request: NextRequest) {
  try {
    const decoded = getUserFromToken(request)

    if (!decoded) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      )
    }

    const body = await request.json()
    const { fullName, email, company, role } = body

    let updatedUser
    let dbType = 'In-Memory'

    try {
      // Try MongoDB first
      updatedUser = await prisma.user.update({
        where: { id: decoded.userId },
        data: {
          fullName,
          email,
          company,
          role,
        },
      })
      
      dbType = 'MongoDB'
      console.log('✅ User profile updated in MongoDB')
    } catch (mongoError) {
      console.warn('⚠️ MongoDB unavailable, updating in-memory storage')
      
      // Fallback to in-memory
      updatedUser = await inMemoryDB.updateUser(decoded.userId, {
        fullName,
        email,
        company,
        role,
      })

      if (!updatedUser) {
        return NextResponse.json(
          { error: 'User not found' },
          { status: 404 }
        )
      }
    }

    const { password: _, ...userWithoutPassword } = updatedUser

    return NextResponse.json({ 
      user: userWithoutPassword,
      databaseType: dbType 
    })
  } catch (error) {
    console.error('Update user error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}
