# User Data Persistence

## Overview

The application now saves all floor plan data (uploaded images, walls, and cameras) per user account. Each user's data is stored separately and automatically loaded when they log in.

## What Gets Saved

For each user account, the system saves:

1. **Floor Plan Image**: The uploaded reference image
2. **Wall Segments**: All traced walls with their positions and floor numbers
3. **Camera Nodes**: All placed cameras with their:
   - Label/name
   - Stream URL
   - Position (x, y, z coordinates)
   - Rotation
   - Floor number
   - Status (normal, warning, incident, inactive)
   - Active state

## How It Works

### Automatic Saving

Data is automatically saved to the server whenever you:
- Upload a new floor plan
- Add a camera
- Clear cameras
- Clear walls
- Auto-trace walls from an image

### Automatic Loading

When you log in or refresh the page:
- Your saved floor plan is automatically loaded
- All your cameras are restored to their positions
- All your walls are rendered

### Per-User Storage

Each user has their own isolated data:
- User A's floor plan and cameras are completely separate from User B's
- Logging in with different accounts shows different data
- No data is shared between accounts

## API Endpoints

### GET `/api/user/data`
Loads the current user's floor plan data.

**Headers:**
```
Authorization: Bearer <jwt_token>
```

**Response:**
```json
{
  "cameras": [...],
  "walls": [...],
  "floorPlan": {
    "referenceImage": "data:image/png;base64,..."
  }
}
```

### POST `/api/user/data`
Saves the current user's floor plan data.

**Headers:**
```
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

**Body:**
```json
{
  "cameras": [...],
  "walls": [...],
  "referenceImage": "data:image/png;base64,..."
}
```

## Database

Currently using an **in-memory fallback database** for development.

### Data Structure

**Camera:**
- `id`: Unique identifier
- `userId`: Owner's user ID
- `label`: Camera name
- `streamUrl`: Video stream URL
- `position`: JSON string of [x, y, z]
- `rotation`: JSON string of [x, y, z]
- `floor`: Floor number (1-5)
- `status`: 'normal' | 'warning' | 'incident' | 'inactive'
- `active`: Boolean

**WallSegment:**
- `id`: Unique identifier
- `userId`: Owner's user ID
- `floor`: Floor number
- `start`: JSON string of [x, y, z]
- `end`: JSON string of [x, y, z]

**FloorPlan:**
- `id`: Unique identifier
- `userId`: Owner's user ID
- `referenceImage`: Base64 encoded image or null

## Migration to MongoDB (Future)

To switch to persistent MongoDB storage:

1. Ensure MongoDB is running on `mongodb://localhost:27017/`
2. Update `frontend/lib/prisma.ts` to initialize PrismaClient
3. Update API routes to use Prisma instead of `inMemoryDB`
4. Run `npx prisma db push` to sync the schema

The schema is already defined in `frontend/prisma/schema.prisma` and ready for MongoDB.

## Testing

1. **Create an account** at `/signup`
2. **Upload a floor plan** in the dashboard
3. **Place some cameras**
4. **Log out** (Profile page)
5. **Log back in** - your floor plan and cameras should be restored

## Notes

- Data is saved automatically, no manual save button needed
- The in-memory database resets when the dev server restarts
- For production, use MongoDB for persistent storage
- JWT tokens are stored in localStorage and used for authentication

