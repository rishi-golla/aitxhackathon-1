import { NextRequest, NextResponse } from 'next/server'
import { prisma } from '@/lib/prisma'
import { inMemoryDB } from '@/lib/db-fallback'
import bcrypt from 'bcryptjs'
import jwt from 'jsonwebtoken'

const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key-change-in-production'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { email, password, fullName, company, role } = body

    // Validation
    if (!email || !password || !fullName || !company) {
      return NextResponse.json(
        { error: 'Missing required fields' },
        { status: 400 }
      )
    }

    if (password.length < 6) {
      return NextResponse.json(
        { error: 'Password must be at least 6 characters' },
        { status: 400 }
      )
    }

    // Hash password
    const hashedPassword = await bcrypt.hash(password, 10)

    let user
    let dbType = 'In-Memory'

    try {
      // Try MongoDB/Prisma first
      const existingUser = await prisma.user.findUnique({ where: { email } })
      
      if (existingUser) {
        return NextResponse.json(
          { error: 'Email already registered' },
          { status: 400 }
        )
      }

      user = await prisma.user.create({
        data: {
          email,
          password: hashedPassword,
          fullName,
          company,
          role: role || 'operator',
          avatar: `https://api.dicebear.com/7.x/initials/svg?seed=${fullName}`,
        },
      })
      
      dbType = 'MongoDB'
      console.log('✅ User created in MongoDB')
    } catch (mongoError) {
      console.warn('⚠️ MongoDB unavailable, using in-memory storage')
      
      // Fallback to in-memory
      const existingUser = await inMemoryDB.findUserByEmail(email)
      
      if (existingUser) {
        return NextResponse.json(
          { error: 'Email already registered' },
          { status: 400 }
        )
      }

      user = await inMemoryDB.createUser({
        email,
        password: hashedPassword,
        fullName,
        company,
        role: role || 'operator',
        avatar: `https://api.dicebear.com/7.x/initials/svg?seed=${fullName}`,
      })
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
    console.error('Signup error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}

