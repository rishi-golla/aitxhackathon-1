# Default Layout Setup ✅

## ✅ Your Floor Plan is Now Hardcoded!

Your uploaded floor plan (`download.jpg`) is now the **permanent default layout** that everyone sees when they open the app!

---

## 🎯 What's Included

### 1. **Your Floor Plan**
- File: `download.jpg` (converted to base64)
- Embedded in: `frontend/lib/hardcoded-data.ts`
- **Loads automatically** for everyone!

### 2. **Pre-Placed Cameras**
4 cameras are already placed on the layout:

| Camera | Label | Position | Status |
|--------|-------|----------|--------|
| `cam-entrance` | Main Entrance | Front area | Normal ✅ |
| `cam-warehouse-1` | Warehouse Bay 1 | Left side | Normal ✅ |
| `cam-warehouse-2` | Warehouse Bay 2 | Right side | Normal ✅ |
| `cam-loading-dock` | Loading Dock | Back area | Warning ⚠️ |

### 3. **Auto-Trace Walls**
- Walls array is empty by default
- Will auto-trace from your floor plan image
- Or you can manually add walls

---

## 🚀 How It Works

### When Anyone Opens the App:

1. **Floor plan loads** → Your `download.jpg` appears
2. **4 cameras appear** → Pre-placed at default positions
3. **Walls auto-trace** → From the floor plan image
4. **Ready to use** → No setup required!

---

## 📝 Adjusting Camera Positions

If you want to change where the cameras are placed, edit `frontend/lib/hardcoded-data.ts`:

```typescript
export const HARDCODED_CAMERAS: Camera[] = [
  {
    id: 'cam-entrance',
    label: 'Main Entrance',
    streamUrl: 'rtsp://demo.com/stream1',
    floor: 1,
    position: JSON.stringify([3, 0, 4]), // [x, y, z] - EDIT THESE
    rotation: JSON.stringify([0, 0, 0]), // [x, y, z] rotation
    status: 'normal',
    active: true,
  },
  // ... more cameras
]
```

### Finding Camera Positions:

1. **Run the app** and place cameras where you want them
2. **Get current positions**:
   ```powershell
   Invoke-WebRequest -Uri "http://localhost:3000/api/user/data" | Select-Object -ExpandProperty Content
   ```
3. **Copy the positions** from the output
4. **Paste into `hardcoded-data.ts`**
5. **Commit to GitHub**

---

## 🎥 Adding More Cameras

Add to the `HARDCODED_CAMERAS` array:

```typescript
{
  id: 'cam-5',
  label: 'Office Area',
  streamUrl: 'rtsp://demo.com/stream5',
  floor: 1,
  position: JSON.stringify([5, 0, 2]),
  rotation: JSON.stringify([0, Math.PI, 0]),
  status: 'normal',
  active: true,
},
```

---

## 🗺️ Changing the Floor Plan

To use a different floor plan:

### Option 1: Replace the Image
1. Replace `download.jpg` with your new image
2. Run this script:
   ```powershell
   cd c:\Users\rishi\aitxhackathon-1-1
   $base64 = [Convert]::ToBase64String([System.IO.File]::ReadAllBytes("download.jpg"))
   $dataUrl = "data:image/jpeg;base64,$base64"
   # Update hardcoded-data.ts with new $dataUrl
   ```

### Option 2: Use a URL
```typescript
export const HARDCODED_FLOORPLAN: FloorPlan = {
  referenceImage: 'https://example.com/your-floor-plan.png',
}
```

---

## 👥 For Teammates

When teammates clone the repo:

```bash
git clone <repo-url>
cd aitxhackathon-1-1
npm install && cd frontend && npm install && cd ..
npm run dev
```

**They see**:
- ✅ Your floor plan
- ✅ Your 4 pre-placed cameras
- ✅ Auto-traced walls
- ✅ Everything ready to go!

---

## 📊 Current Setup

```typescript
Floor Plan: download.jpg (embedded as base64)
Cameras: 4 pre-placed
  - Main Entrance (normal)
  - Warehouse Bay 1 (normal)
  - Warehouse Bay 2 (normal)
  - Loading Dock (warning)
Walls: Auto-traced from image
```

---

## ✅ Benefits

✅ **Instant Setup** - No configuration needed  
✅ **Same for Everyone** - Exact same layout  
✅ **Version Controlled** - In Git  
✅ **No Database** - All in code  
✅ **Fast** - Loads instantly  
✅ **Professional** - Pre-configured demo  

---

## 🔄 Updating the Default Layout

### Step 1: Design in App
1. Run `npm run dev`
2. Adjust camera positions
3. Add more cameras if needed

### Step 2: Export Data
```powershell
Invoke-WebRequest -Uri "http://localhost:3000/api/user/data" > layout.json
```

### Step 3: Update Code
Copy camera positions from `layout.json` to `hardcoded-data.ts`

### Step 4: Commit
```bash
git add frontend/lib/hardcoded-data.ts
git commit -m "Update default camera positions"
git push
```

### Step 5: Teammates Pull
```bash
git pull
npm run dev
```

**They see your updated layout!** ✅

---

## 📁 Files Modified

- ✅ `frontend/lib/hardcoded-data.ts` - Contains your floor plan + cameras
- ✅ `download.jpg` - Your original floor plan image
- ✅ All API routes use hardcoded data

---

## 🎉 Summary

**What You Wanted**: Floor plan and cameras hardcoded for everyone  
**What You Got**: 
- ✅ Your `download.jpg` embedded in code
- ✅ 4 cameras pre-placed
- ✅ Loads automatically for everyone
- ✅ No database, no setup

**Status**: COMPLETE! 🚀

---

## 🧪 Test It

1. **Open the app** → Floor plan should appear
2. **See 4 cameras** → Already placed
3. **Walls auto-trace** → From your image
4. **Close and reopen** → Same layout
5. **Have teammate clone** → They see same thing!

---

**Your default layout is now live!** Anyone who clones the repo will see your floor plan with 4 cameras already placed! 🎉

