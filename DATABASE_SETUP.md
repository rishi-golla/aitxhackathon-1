# Database Setup Guide

## Overview
This application uses **MongoDB** with **Prisma ORM** for data persistence. The database stores users, cameras, accidents, and wall segments.

## Database Schema

### Tables

#### 1. **User**
Stores user account information.
- `id` - Unique identifier (CUID)
- `email` - User email (unique)
- `password` - Hashed password (bcrypt)
- `fullName` - User's full name
- `company` - Company name
- `role` - User role (operator, supervisor, manager, admin)
- `avatar` - Avatar URL
- `createdAt` - Account creation timestamp
- `updatedAt` - Last update timestamp

#### 2. **Camera**
Stores camera configuration and placement.
- `id` - Unique identifier
- `label` - Camera name (e.g., "Camera N°2")
- `streamUrl` - Video stream URL
- `floor` - Floor number (1-5)
- `position` - 3D position as JSON string [x, y, z]
- `rotation` - 3D rotation as JSON string [x, y, z]
- `status` - Status (normal, warning, incident, inactive)
- `active` - Boolean active state
- `createdAt` - Creation timestamp
- `updatedAt` - Last update timestamp

#### 3. **Accident**
Stores accident incident records.
- `id` - Unique identifier
- `type` - Incident type (crowd, smoke, fall, fire, collision, spill)
- `severity` - Severity level (low, medium, high, critical)
- `title` - Incident title
- `description` - Detailed description
- `cameraId` - Associated camera ID
- `cameraName` - Camera name
- `floor` - Floor number
- `timestamp` - Incident timestamp
- `duration` - Video duration
- `videoUrl` - Video file URL
- `status` - Status (new, reviewing, resolved, archived)
- `assignedTo` - Assigned team/person
- `createdAt` - Record creation timestamp
- `updatedAt` - Last update timestamp

#### 4. **WallSegment**
Stores wall segments for floor plans.
- `id` - Unique identifier
- `floor` - Floor number
- `start` - Start point as JSON string [x, y]
- `end` - End point as JSON string [x, y]
- `createdAt` - Creation timestamp

## Setup Instructions

### 1. Install Dependencies
```bash
cd frontend
npm install prisma @prisma/client bcryptjs jsonwebtoken --legacy-peer-deps
npm install --save-dev @types/bcryptjs @types/jsonwebtoken --legacy-peer-deps
```

### 2. Initialize Database
The database is already initialized. To reset or recreate:
```bash
cd frontend
npx prisma db push
```

**Note:** MongoDB with Prisma doesn't use migrations. Use `db push` instead.

### 3. Generate Prisma Client
```bash
cd frontend
npx prisma generate
```

### 4. View Database
Use Prisma Studio to view/edit data:
```bash
cd frontend
npx prisma studio
```
This opens a web interface at `http://localhost:5555`

## API Endpoints

### Authentication

#### POST `/api/auth/signup`
Create a new user account.
```json
{
  "email": "user@example.com",
  "password": "password123",
  "fullName": "John Doe",
  "company": "Acme Corp",
  "role": "operator"
}
```

#### POST `/api/auth/login`
Login with email and password.
```json
{
  "email": "user@example.com",
  "password": "password123"
}
```

#### GET `/api/auth/me`
Get current user profile (requires Bearer token).

#### PUT `/api/auth/me`
Update user profile (requires Bearer token).
```json
{
  "fullName": "John Doe",
  "email": "newemail@example.com",
  "company": "New Company",
  "role": "supervisor"
}
```

## Environment Variables

Create a `.env` file in the `frontend` directory:
```env
DATABASE_URL="mongodb://localhost:27017/nda-company"
JWT_SECRET="your-secret-key-change-in-production"
```

**Prerequisites:**
- MongoDB must be installed and running on your system
- Default MongoDB port: 27017
- Database name: `nda-company`

### Install MongoDB:
- **Windows:** Download from [mongodb.com](https://www.mongodb.com/try/download/community)
- **Mac:** `brew install mongodb-community`
- **Linux:** `sudo apt-get install mongodb`

### Start MongoDB:
```bash
# Windows (if installed as service)
net start MongoDB

# Mac/Linux
mongod
```

## Security Features

1. **Password Hashing**: Uses bcrypt with salt rounds of 10
2. **JWT Authentication**: 7-day expiration tokens
3. **Protected Routes**: API endpoints validate JWT tokens
4. **SQL Injection Protection**: Prisma ORM provides parameterized queries

## Database Location

MongoDB stores data in its default data directory:
- **Windows:** `C:\data\db`
- **Mac/Linux:** `/data/db`

Database name: `nda-company`

## Schema Updates

MongoDB with Prisma doesn't use migrations. After schema changes:
```bash
cd frontend
npx prisma generate
npx prisma db push
```

## Backup

To backup the database:
```bash
# Copy the database file
cp frontend/dev.db frontend/dev.db.backup
```

## Production Considerations

For production, consider:
1. Use MongoDB Atlas (cloud-hosted) instead of local MongoDB
2. Change `JWT_SECRET` to a strong random value
3. Enable HTTPS
4. Implement rate limiting
5. Add database connection pooling
6. Set up automated backups
7. Use environment-specific configurations
8. Enable MongoDB authentication
9. Use replica sets for high availability

## Troubleshooting

### Reset Database
```bash
# Drop all collections
mongosh nda-company --eval "db.dropDatabase()"

# Push schema again
cd frontend
npx prisma db push
```

### View Data with Prisma Studio
```bash
cd frontend
npx prisma studio
```

### View Data with MongoDB Shell
```bash
mongosh nda-company
db.User.find()
db.Camera.find()
db.Accident.find()
```

### Check MongoDB Connection
```bash
mongosh mongodb://localhost:27017/nda-company
```

