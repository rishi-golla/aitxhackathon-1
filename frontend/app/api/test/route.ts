import { NextResponse } from 'next/server'
import { inMemoryDB } from '@/lib/db-fallback'

export async function GET() {
  try {
    let userCount = 0
    let dbType = 'In-Memory'
    let note = 'Using temporary in-memory storage. Data will be lost on restart. Install MongoDB for persistent storage.'

    try {
      // Try MongoDB first
      const { prisma } = await import('@/lib/prisma')
      userCount = await prisma.user.count()
      dbType = 'MongoDB'
      note = 'Connected to MongoDB. Data persists across restarts.'
      console.log('✅ Test route: MongoDB is working')
    } catch (error) {
      console.warn('⚠️ Test route: MongoDB unavailable, using in-memory storage')
      userCount = await inMemoryDB.countUsers()
    }
    
    return NextResponse.json({
      status: 'ok',
      message: 'API is working',
      database: 'connected',
      databaseType: dbType,
      userCount,
      note,
    })
  } catch (error) {
    console.error('Test route error:', error)
    return NextResponse.json({
      status: 'error',
      message: 'API test failed',
      error: error instanceof Error ? error.message : 'Unknown error',
    }, { status: 500 })
  }
}

