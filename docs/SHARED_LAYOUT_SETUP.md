# Shared Layout Setup - Same Layout for Everyone

## Current Situation

Right now, **each person has their own local MongoDB**, so:
- ❌ You upload a floor plan → Only you see it
- ❌ Teammate clones repo → They see empty layout
- ❌ Changes don't sync between machines

## Goal

Make it so:
- ✅ You upload a floor plan → **Everyone sees it**
- ✅ You place cameras → **Everyone sees them**
- ✅ Any changes → **Instantly shared**

## Solution: Shared Cloud MongoDB

### Option 1: MongoDB Atlas (Free & Recommended)

**MongoDB Atlas** is a free cloud MongoDB service. Everyone connects to the same database.

#### Step 1: Create Free MongoDB Atlas Account
1. Go to https://www.mongodb.com/cloud/atlas/register
2. Sign up (free tier is perfect)
3. Create a free cluster (takes 3-5 minutes)

#### Step 2: Get Connection String
1. In Atlas, click "Connect"
2. Choose "Connect your application"
3. Copy the connection string:
   ```
   mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/nda-company
   ```

#### Step 3: Update `.env` File
```env
# Replace localhost with your Atlas connection string
DATABASE_URL="mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/nda-company"
JWT_SECRET="your-secret-key-change-in-production"
```

#### Step 4: Share Connection String
**Add to `.env.example`** (safe to commit):
```env
# .env.example
DATABASE_URL="mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/nda-company"
JWT_SECRET="your-secret-key-change-in-production"
```

**Tell teammates**: "Copy `.env.example` to `.env`"

#### Step 5: Everyone Restarts
```powershell
npm run dev
```

Now **everyone sees the same layout!** 🎉

---

### Option 2: Local Network MongoDB (Same WiFi)

If everyone is on the **same WiFi network**, you can share your local MongoDB:

#### Step 1: Find Your IP Address
```powershell
ipconfig | findstr IPv4
```
Example output: `192.168.1.100`

#### Step 2: Configure MongoDB to Accept Remote Connections
Edit MongoDB config file (usually `C:\Program Files\MongoDB\Server\8.0\bin\mongod.cfg`):
```yaml
net:
  bindIp: 0.0.0.0  # Allow connections from any IP
  port: 27017
```

Restart MongoDB service.

#### Step 3: Teammates Update `.env`
```env
# Replace localhost with your IP
DATABASE_URL="mongodb://192.168.1.100:27017/nda-company"
```

#### Step 4: Everyone Restarts
```powershell
npm run dev
```

**Limitation**: Only works when on same WiFi.

---

### Option 3: GitHub + MongoDB Atlas (Best for Teams)

**Recommended workflow**:

1. **Use MongoDB Atlas** (free cloud database)
2. **Add connection string to `.env.example`**:
   ```env
   DATABASE_URL="mongodb+srv://team:password@cluster.mongodb.net/nda-company"
   ```
3. **Commit `.env.example` to GitHub**
4. **Add `.env` to `.gitignore`** (already done)

**When teammate clones repo**:
```bash
git clone <repo>
cd aitxhackathon-1-1
cp frontend/.env.example frontend/.env  # Copy example to actual .env
npm run dev
```

They automatically connect to the shared database!

---

## Current Setup (Already Done)

✅ **Shared Demo User**: Everyone uses `demo-user-id`  
✅ **Same Collections**: Camera, WallSegment, FloorPlan  
✅ **Same Database Name**: `nda-company`  

**Only missing**: Shared MongoDB connection!

---

## Quick Start (MongoDB Atlas)

### 1. Create Atlas Account (5 minutes)
- https://www.mongodb.com/cloud/atlas/register
- Create free cluster
- Get connection string

### 2. Update `.env`
```env
DATABASE_URL="mongodb+srv://YOUR_CONNECTION_STRING"
```

### 3. Restart App
```powershell
npm run dev
```

### 4. Test It
- Upload a floor plan
- Place cameras
- Have teammate run `npm run dev`
- They see your layout! ✅

---

## Verification

After setup, verify it works:

**On your laptop**:
```powershell
cd c:\Users\rishi\aitxhackathon-1-1
node check-user.js
```

**On teammate's laptop**:
```powershell
cd /path/to/aitxhackathon-1-1
node check-user.js
```

Both should show the **same cameras and walls**!

---

## Benefits

✅ **Real-time sync** - Changes appear for everyone  
✅ **No manual sharing** - Just clone and run  
✅ **Cloud backup** - Data safe even if laptop dies  
✅ **Free** - MongoDB Atlas free tier is generous  
✅ **Team collaboration** - Everyone sees same layout  

---

## Current Status

- ✅ App uses demo user (no auth)
- ✅ Data structure ready for sharing
- 🔄 **Need**: Shared MongoDB connection

**Next Step**: Set up MongoDB Atlas (5 minutes) and everyone will see the same layout!

