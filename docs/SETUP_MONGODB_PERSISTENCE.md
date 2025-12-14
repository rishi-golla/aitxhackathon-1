# 🗄️ MongoDB Persistent Storage Setup

## ✅ Code is Ready!

Your application code is now configured to use MongoDB for persistent storage with automatic fallback to in-memory storage.

---

## 📋 Step-by-Step Setup

### Step 1: Install MongoDB

#### **Option A: Using Chocolatey (Run PowerShell as Administrator)**
```powershell
# Right-click PowerShell → "Run as Administrator"
choco install mongodb -y
```

#### **Option B: Manual Download**
1. Download: https://www.mongodb.com/try/download/community
2. Run installer as Administrator
3. Choose "Complete" installation
4. ✅ Check "Install MongoDB as a Service"
5. Use default settings

---

### Step 2: Verify MongoDB is Running

```powershell
# Check MongoDB service status
Get-Service -Name MongoDB

# Should show: Status = Running
```

If not running, start it:
```powershell
net start MongoDB
```

---

### Step 3: Generate Prisma Client

Once MongoDB is installed and running:

```powershell
cd frontend
npx prisma generate
npx prisma db push
```

This will:
- Generate the Prisma client
- Create the database collections in MongoDB

---

### Step 4: Restart Your App

```powershell
# Kill existing processes
taskkill /F /IM node.exe

# Clear cache
cd frontend
Remove-Item -Recurse -Force .next

# Start fresh
cd ..
npm run dev
```

---

## 🎯 How It Works Now

### **Automatic Database Selection**

Your app now tries MongoDB first, then falls back to in-memory:

```
1. Try MongoDB → ✅ Success → Data persists forever
2. MongoDB fails → ⚠️ Fallback → In-memory (resets on restart)
```

### **What Gets Saved to MongoDB**

- ✅ **Users** → `User` collection
- ✅ **Cameras** → `Camera` collection
- ✅ **Walls** → `WallSegment` collection
- ✅ **Floor Plans** → `FloorPlan` collection

### **Database Name**

```
mongodb://localhost:27017/nda-company
```

---

## 🔍 Verify It's Working

### Check Console Logs

When you sign up or save data, you should see:

```
✅ User created in MongoDB
✅ Data saved to MongoDB
✅ Data loaded from MongoDB
```

Instead of:

```
⚠️ MongoDB unavailable, using in-memory storage
```

### Check MongoDB Directly

```powershell
# Connect to MongoDB (after installing mongosh separately if needed)
mongosh mongodb://localhost:27017/nda-company

# List collections
show collections

# Check users
db.User.find()

# Check cameras
db.Camera.find()
```

---

## 📊 Database Collections

### `User`
```json
{
  "_id": ObjectId("..."),
  "email": "user@example.com",
  "password": "hashed...",
  "fullName": "John Doe",
  "company": "ACME Corp",
  "role": "operator",
  "avatar": "https://...",
  "createdAt": ISODate("..."),
  "updatedAt": ISODate("...")
}
```

### `Camera`
```json
{
  "_id": ObjectId("..."),
  "userId": "user-id",
  "label": "Camera 1",
  "streamUrl": "rtsp://...",
  "floor": 1,
  "position": "[1.5, 0.15, 2.3]",
  "rotation": "[0, 0, 0]",
  "status": "normal",
  "active": true,
  "createdAt": ISODate("..."),
  "updatedAt": ISODate("...")
}
```

### `WallSegment`
```json
{
  "_id": ObjectId("..."),
  "userId": "user-id",
  "floor": 1,
  "start": "[0, 0]",
  "end": "[5, 0]",
  "createdAt": ISODate("...")
}
```

### `FloorPlan`
```json
{
  "_id": ObjectId("..."),
  "userId": "user-id",
  "referenceImage": "data:image/png;base64,...",
  "createdAt": ISODate("..."),
  "updatedAt": ISODate("...")
}
```

---

## 🚨 Troubleshooting

### "MongoDB service not found"
MongoDB is not installed. Follow Step 1 again.

### "Connection refused"
MongoDB is not running:
```powershell
net start MongoDB
```

### "Port 27017 in use"
Another process is using the port:
```powershell
netstat -ano | findstr :27017
# Kill the process if needed
```

### Still using in-memory after installation
1. Make sure MongoDB is running: `Get-Service -Name MongoDB`
2. Run `npx prisma generate` in the `frontend` folder
3. Run `npx prisma db push` in the `frontend` folder
4. Restart your app completely

---

## 🎉 Benefits of MongoDB

✅ **Persistent Storage** - Data survives server restarts
✅ **Per-User Data** - Each account has isolated data
✅ **Scalable** - Can handle thousands of users
✅ **Queryable** - Easy to search and analyze data
✅ **Backup-able** - Can export/import data easily

---

## 📝 Next Steps After Setup

1. **Install MongoDB** (Step 1)
2. **Verify it's running** (Step 2)
3. **Generate Prisma client** (Step 3)
4. **Restart app** (Step 4)
5. **Create an account** - Should see "✅ User created in MongoDB"
6. **Upload floor plan & add cameras** - Should see "✅ Data saved to MongoDB"
7. **Restart server** - Your data should still be there! 🎊

---

## ⚡ Quick Command Reference

```powershell
# Check MongoDB status
Get-Service -Name MongoDB

# Start MongoDB
net start MongoDB

# Stop MongoDB
net stop MongoDB

# Generate Prisma client
cd frontend
npx prisma generate

# Push schema to MongoDB
npx prisma db push

# View MongoDB data
mongosh mongodb://localhost:27017/nda-company
```

