# MongoDB Installation & Setup Guide for Windows

## Step 1: Install MongoDB Community Server

### Option A: Using Winget (Recommended - Fastest)
```powershell
winget install MongoDB.Server
```

### Option B: Manual Download
1. Go to: https://www.mongodb.com/try/download/community
2. Select:
   - Version: 8.0.4 (or latest)
   - Platform: Windows
   - Package: MSI
3. Download and run the installer
4. During installation:
   - Choose "Complete" installation
   - ✅ Check "Install MongoDB as a Service"
   - ✅ Check "Run service as Network Service user"
   - Use default data directory: `C:\Program Files\MongoDB\Server\8.0\data`

## Step 2: Verify Installation

After installation, verify MongoDB is running:

```powershell
# Check if MongoDB service is running
Get-Service -Name MongoDB

# Should show: Status = Running
```

## Step 3: Test Connection

```powershell
# Try connecting (mongosh should now work)
mongosh mongodb://localhost:27017/

# You should see:
# Current Mongosh Log ID: ...
# Connecting to: mongodb://localhost:27017/
# Using MongoDB: ...
```

## Step 4: Create Database and User (Optional but Recommended)

```javascript
// In mongosh:
use nda-company

// Create a user for the database
db.createUser({
  user: "ndauser",
  pwd: "your-secure-password",
  roles: [{ role: "readWrite", db: "nda-company" }]
})
```

## Step 5: Update Your App Configuration

Once MongoDB is installed and running, I'll update your app to use it instead of in-memory storage.

---

## Quick Start (After Installation)

1. **Install MongoDB** using one of the methods above
2. **Verify it's running**: `Get-Service -Name MongoDB`
3. **Let me know** and I'll update the code to use MongoDB
4. **Your data will persist** across server restarts! 🎉

---

## Troubleshooting

### MongoDB Service Won't Start
```powershell
# Start MongoDB manually
net start MongoDB

# Or restart it
net stop MongoDB
net start MongoDB
```

### Port 27017 Already in Use
```powershell
# Check what's using port 27017
netstat -ano | findstr :27017

# Kill the process if needed (replace PID with actual process ID)
taskkill /PID <PID> /F
```

### Can't Connect to MongoDB
```powershell
# Check if MongoDB is listening
netstat -an | findstr :27017

# Should show: TCP 0.0.0.0:27017 LISTENING
```

---

## Alternative: MongoDB Atlas (Cloud - No Installation Needed)

If you prefer not to install MongoDB locally:

1. Go to: https://www.mongodb.com/cloud/atlas/register
2. Create a free account
3. Create a free cluster (M0)
4. Get your connection string
5. I'll update the app to use your Atlas connection string

Connection string format:
```
mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/nda-company?retryWrites=true&w=majority
```

