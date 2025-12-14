# Quick Start - Get Running in 2 Minutes

## For New Team Members

### 1. Clone the Repo
```bash
git clone <repo-url>
cd aitxhackathon-1-1
```

### 2. Copy Environment File
```bash
# Windows
cd frontend
copy .env.example .env
cd ..

# Mac/Linux
cd frontend
cp .env.example .env
cd ..
```

### 3. Install Dependencies
```bash
npm install
cd frontend
npm install
cd ..
```

### 4. Run the App
```bash
npm run dev
```

**That's it!** The Electron app opens with the dashboard. 🎉

---

## What You'll See

- ✅ Dashboard with camera layout
- ✅ Cost of Inaction metrics
- ✅ Recent Activity feed
- ✅ Accidents tracking
- ✅ **Same layout as everyone else** (if using shared MongoDB)

---

## Shared Layout

If the project owner has set up MongoDB Atlas:
- ✅ You see their floor plan
- ✅ You see their camera placements
- ✅ Any changes you make are visible to everyone
- ✅ Real-time collaboration!

If using local MongoDB:
- Each person has their own layout
- See `TEAM_SETUP_GUIDE.md` to enable sharing

---

## Features

- 📸 **Upload Floor Plans** - Drag & drop factory layouts
- 🎥 **Place Cameras** - Click to add camera nodes
- 🗺️ **Auto-Trace Walls** - Automatic wall detection
- 💰 **Cost of Inaction** - Real-time violation cost tracking
- 📊 **Analytics** - Accidents, incidents, metrics
- 🔴 **Live Feeds** - View all camera feeds
- 📝 **Activity Log** - Recent events and alerts

---

## Troubleshooting

### Port 3000 already in use
```powershell
taskkill /F /IM node.exe
npm run dev
```

### White screen / errors
```powershell
cd frontend
Remove-Item -Recurse -Force .next
cd ..
npm run dev
```

### MongoDB connection error
Check `frontend/.env` file exists and has valid `DATABASE_URL`

---

## Project Structure

```
aitxhackathon-1-1/
├── backend/           # Electron main process
├── frontend/          # Next.js app
│   ├── app/          # Pages and routes
│   ├── lib/          # MongoDB connection
│   └── .env          # Your local config
├── docs/             # Documentation
└── package.json      # Root dependencies
```

---

## Commands

```bash
npm run dev           # Start dev server + Electron
npm run dev:next      # Start Next.js only
npm run dev:electron  # Start Electron only
npm run build:next    # Build for production
```

---

## Need Help?

- See `TEAM_SETUP_GUIDE.md` for shared MongoDB setup
- See `NO_AUTH_SETUP.md` for authentication details
- See `SHARED_LAYOUT_SETUP.md` for collaboration setup

---

**Ready to go!** 🚀
