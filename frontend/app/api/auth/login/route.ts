import { NextRequest, NextResponse } from 'next/server'
import { getDb } from '@/lib/mongodb'
import { inMemoryDB } from '@/lib/db-fallback'
import bcrypt from 'bcryptjs'
import jwt from 'jsonwebtoken'

const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key-change-in-production'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { email, password } = body

    // Validation
    if (!email || !password) {
      return NextResponse.json(
        { error: 'Email and password are required' },
        { status: 400 }
      )
    }

    let user
    let dbType = 'In-Memory'

    try {
      // Try MongoDB directly
      const db = await getDb()
      const usersCollection = db.collection('User')
      
      const mongoUser = await usersCollection.findOne({ email })
      
      if (mongoUser) {
        user = {
          id: mongoUser._id.toString(),
          email: mongoUser.email,
          password: mongoUser.password,
          fullName: mongoUser.fullName,
          company: mongoUser.company,
          role: mongoUser.role,
          avatar: mongoUser.avatar,
          createdAt: mongoUser.createdAt,
          updatedAt: mongoUser.updatedAt,
        }
        dbType = 'MongoDB'
        console.log('✅ User found in MongoDB')
      }
    } catch (mongoError) {
      console.warn('⚠️ MongoDB unavailable, checking in-memory storage:', mongoError)
    }

    // Fallback to in-memory if not found in MongoDB
    if (!user) {
      user = await inMemoryDB.findUserByEmail(email)
    }

    if (!user) {
      return NextResponse.json(
        { error: 'Invalid email or password' },
        { status: 401 }
      )
    }

    // Verify password
    const isPasswordValid = await bcrypt.compare(password, user.password)

    if (!isPasswordValid) {
      return NextResponse.json(
        { error: 'Invalid email or password' },
        { status: 401 }
      )
    }

    // Generate JWT token
    const token = jwt.sign(
      { userId: user.id, email: user.email },
      JWT_SECRET,
      { expiresIn: '7d' }
    )

    // Return user data (without password)
    const { password: _, ...userWithoutPassword } = user

    return NextResponse.json({
      user: userWithoutPassword,
      token,
      databaseType: dbType,
    })
  } catch (error) {
    console.error('Login error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}

