// Simple in-memory database fallback when MongoDB is not available
interface User {
  id: string
  email: string
  password: string
  fullName: string
  company: string
  role: string
  avatar: string
  createdAt: Date
  updatedAt: Date
}

interface Camera {
  id: string
  userId: string
  label: string
  streamUrl: string
  floor: number
  position: string
  rotation: string
  status: string
  active: boolean
  createdAt: Date
  updatedAt: Date
}

interface WallSegment {
  id: string
  userId: string
  floor: number
  start: string
  end: string
  createdAt: Date
}

interface FloorPlan {
  id: string
  userId: string
  referenceImage: string | null
  createdAt: Date
  updatedAt: Date
}

class InMemoryDB {
  private users: User[] = []
  private cameras: Camera[] = []
  private wallSegments: WallSegment[] = []
  private floorPlans: FloorPlan[] = []

  // User methods
  async createUser(data: Omit<User, 'id' | 'createdAt' | 'updatedAt'>): Promise<User> {
    const user: User = {
      ...data,
      id: Date.now().toString(),
      createdAt: new Date(),
      updatedAt: new Date(),
    }
    this.users.push(user)
    
    // Create default floor plan for user
    this.floorPlans.push({
      id: `fp-${Date.now()}`,
      userId: user.id,
      referenceImage: null,
      createdAt: new Date(),
      updatedAt: new Date(),
    })
    
    return user
  }

  async findUserByEmail(email: string): Promise<User | null> {
    return this.users.find(u => u.email === email) || null
  }

  async findUserById(id: string): Promise<User | null> {
    return this.users.find(u => u.id === id) || null
  }

  async updateUser(id: string, data: Partial<User>): Promise<User | null> {
    const index = this.users.findIndex(u => u.id === id)
    if (index === -1) return null
    
    this.users[index] = {
      ...this.users[index],
      ...data,
      updatedAt: new Date(),
    }
    return this.users[index]
  }

  async countUsers(): Promise<number> {
    return this.users.length
  }

  // Camera methods
  async createCamera(userId: string, data: Omit<Camera, 'id' | 'userId' | 'createdAt' | 'updatedAt'>): Promise<Camera> {
    const camera: Camera = {
      ...data,
      id: `cam-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      userId,
      createdAt: new Date(),
      updatedAt: new Date(),
    }
    this.cameras.push(camera)
    return camera
  }

  async findCamerasByUserId(userId: string): Promise<Camera[]> {
    return this.cameras.filter(c => c.userId === userId)
  }

  async deleteCamera(id: string, userId: string): Promise<boolean> {
    const index = this.cameras.findIndex(c => c.id === id && c.userId === userId)
    if (index === -1) return false
    this.cameras.splice(index, 1)
    return true
  }

  async deleteCamerasByUserId(userId: string): Promise<void> {
    this.cameras = this.cameras.filter(c => c.userId !== userId)
  }

  // Wall segment methods
  async createWallSegment(userId: string, data: Omit<WallSegment, 'id' | 'userId' | 'createdAt'>): Promise<WallSegment> {
    const wall: WallSegment = {
      ...data,
      id: `wall-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      userId,
      createdAt: new Date(),
    }
    this.wallSegments.push(wall)
    return wall
  }

  async findWallSegmentsByUserId(userId: string): Promise<WallSegment[]> {
    return this.wallSegments.filter(w => w.userId === userId)
  }

  async deleteWallSegmentsByUserId(userId: string): Promise<void> {
    this.wallSegments = this.wallSegments.filter(w => w.userId !== userId)
  }

  // Floor plan methods
  async updateFloorPlan(userId: string, referenceImage: string | null): Promise<FloorPlan> {
    let floorPlan = this.floorPlans.find(fp => fp.userId === userId)
    
    if (!floorPlan) {
      floorPlan = {
        id: `fp-${Date.now()}`,
        userId,
        referenceImage,
        createdAt: new Date(),
        updatedAt: new Date(),
      }
      this.floorPlans.push(floorPlan)
    } else {
      floorPlan.referenceImage = referenceImage
      floorPlan.updatedAt = new Date()
    }
    
    return floorPlan
  }

  async findFloorPlanByUserId(userId: string): Promise<FloorPlan | null> {
    return this.floorPlans.find(fp => fp.userId === userId) || null
  }

  // Get all user data
  async getUserData(userId: string) {
    return {
      cameras: await this.findCamerasByUserId(userId),
      walls: await this.findWallSegmentsByUserId(userId),
      floorPlan: await this.findFloorPlanByUserId(userId),
    }
  }
}

export const inMemoryDB = new InMemoryDB()

