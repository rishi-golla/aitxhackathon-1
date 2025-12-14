# Team Setup Guide - Shared Layout for Everyone

## 🎯 Goal
Make it so **everyone who clones this repo sees the same floor plan and camera layout**.

## ✅ Already Done
- ✅ App uses shared demo user (`demo-user-id`)
- ✅ No authentication required
- ✅ `.env.example` created with instructions
- ✅ Code ready for shared database

## 🚀 Quick Setup (5 Minutes)

### For You (Project Owner):

#### 1. Create Free MongoDB Atlas Account
```
https://www.mongodb.com/cloud/atlas/register
```
- Sign up (free)
- Create a free cluster (M0 Sandbox)
- Wait 3-5 minutes for cluster to deploy

#### 2. Get Connection String
1. Click "Connect" on your cluster
2. Choose "Connect your application"
3. Select "Node.js" driver
4. Copy the connection string:
   ```
   mongodb+srv://username:<password>@cluster0.xxxxx.mongodb.net/
   ```
5. Replace `<password>` with your actual password
6. Add database name at the end: `/nda-company`

Final string looks like:
```
mongodb+srv://myuser:mypassword@cluster0.abc123.mongodb.net/nda-company
```

#### 3. Update Your Local `.env`
```powershell
cd c:\Users\rishi\aitxhackathon-1-1\frontend
notepad .env
```

Replace with your Atlas connection string:
```env
DATABASE_URL="mongodb+srv://myuser:mypassword@cluster0.abc123.mongodb.net/nda-company"
JWT_SECRET="your-secret-key-change-in-production"
```

#### 4. Update `.env.example` for Team
```powershell
notepad .env.example
```

Replace the localhost line with your Atlas connection:
```env
# MongoDB Connection - SHARED FOR TEAM
DATABASE_URL="mongodb+srv://myuser:mypassword@cluster0.abc123.mongodb.net/nda-company"

JWT_SECRET="your-secret-key-change-in-production"
```

#### 5. Commit `.env.example` to GitHub
```bash
git add frontend/.env.example
git commit -m "Add shared MongoDB connection"
git push
```

**Note**: `.env` is already in `.gitignore`, so your local file won't be committed.

#### 6. Restart Your App
```powershell
# Stop current server (Ctrl+C)
npm run dev
```

Your layout now saves to the cloud! ☁️

---

### For Teammates:

#### 1. Clone the Repo
```bash
git clone <repo-url>
cd aitxhackathon-1-1
```

#### 2. Copy `.env.example` to `.env`
```bash
# Windows
cd frontend
copy .env.example .env

# Mac/Linux
cd frontend
cp .env.example .env
```

#### 3. Install Dependencies
```bash
cd ..
npm install
cd frontend
npm install
```

#### 4. Run the App
```bash
cd ..
npm run dev
```

**That's it!** The app opens and shows the **same layout** as everyone else! 🎉

---

## 🧪 Test It Works

### On Your Laptop:
1. Upload a floor plan
2. Place 3 cameras
3. Note their positions

### On Teammate's Laptop:
1. Run `npm run dev`
2. Should see **your floor plan**
3. Should see **your 3 cameras**
4. In exact same positions!

### Verify Database:
```powershell
cd c:\Users\rishi\aitxhackathon-1-1
node check-user.js
```

Both laptops should show the **same data**!

---

## 📊 What Gets Shared

✅ **Floor Plans** - Uploaded images  
✅ **Cameras** - Position, label, floor, status  
✅ **Walls** - Auto-traced or manual walls  
✅ **All Changes** - Real-time sync  

---

## 🔒 Security Note

**For hackathon/demo**: Sharing the connection string in `.env.example` is fine.

**For production**: Use environment variables and don't commit credentials.

---

## 🎯 Current Status

- ✅ Code supports shared database
- ✅ `.env.example` created
- 🔄 **Need**: Set up MongoDB Atlas (5 min)
- 🔄 **Then**: Push `.env.example` to GitHub

---

## 💡 Alternative: Local Network Sharing

If you're all on the **same WiFi** and don't want to use cloud:

### Your Laptop:
```powershell
# Find your IP
ipconfig | findstr IPv4
# Example: 192.168.1.100
```

### Teammates' `.env`:
```env
DATABASE_URL="mongodb://192.168.1.100:27017/nda-company"
```

**Limitation**: Only works on same network.

---

## ✅ Benefits of Shared MongoDB

✅ **Instant Collaboration** - Everyone sees changes  
✅ **No Manual Sync** - Automatic  
✅ **Cloud Backup** - Data safe  
✅ **Easy Onboarding** - Clone and run  
✅ **Free** - MongoDB Atlas free tier  

---

## 📝 Summary

**Right now**: Each person has their own local MongoDB  
**After setup**: Everyone shares one cloud MongoDB  
**Result**: Same layout for everyone! 🎉

**Time to setup**: 5 minutes  
**Cost**: Free  
**Benefit**: Huge!

