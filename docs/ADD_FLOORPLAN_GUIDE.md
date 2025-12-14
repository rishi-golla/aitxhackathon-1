# How to Add Your Floor Plan to Hardcoded Data

## ✅ Error Fixed!

The syntax error has been fixed. The app now runs with 4 pre-placed cameras.

## 🗺️ Adding Your Floor Plan Image

You have 3 options to add your `download.jpg` as the default floor plan:

### Option 1: Upload in the App (Easiest)
1. Run `npm run dev`
2. Click "Upload Floor Plan" in the app
3. Select `download.jpg`
4. The image will be stored in memory while the server runs
5. **Note**: Resets when you restart the server

### Option 2: Use a URL (Recommended)
1. Upload `download.jpg` to an image hosting service (Imgur, GitHub, etc.)
2. Get the direct image URL
3. Edit `frontend/lib/hardcoded-data.ts`:
   ```typescript
   export const HARDCODED_FLOORPLAN: FloorPlan = {
     referenceImage: 'https://your-url.com/download.jpg',
   }
   ```
4. Commit to GitHub
5. Everyone sees the floor plan!

### Option 3: Base64 Embed (For Small Images)
For small images only (< 100KB). Your `download.jpg` is 11KB so it works!

1. Convert to base64:
   ```powershell
   cd c:\Users\rishi\aitxhackathon-1-1
   $base64 = [Convert]::ToBase64String([System.IO.File]::ReadAllBytes("download.jpg"))
   Write-Host "data:image/jpeg;base64,$base64"
   ```

2. Copy the output

3. Edit `frontend/lib/hardcoded-data.ts`:
   ```typescript
   export const HARDCODED_FLOORPLAN: FloorPlan = {
     referenceImage: `data:image/jpeg;base64,YOUR_BASE64_STRING_HERE`,
   }
   ```

4. **Important**: Use backticks (`) not quotes for long strings!

## 🎯 Current Setup

Right now you have:
- ✅ 4 cameras pre-placed
- ✅ Camera positions hardcoded
- ⚠️ Floor plan: `null` (upload in app or use URL)

## 📝 Recommended Approach

**Use Option 2 (URL)**:
1. Upload `download.jpg` to GitHub:
   ```bash
   git add download.jpg
   git commit -m "Add floor plan image"
   git push
   ```

2. Get the raw GitHub URL:
   ```
   https://raw.githubusercontent.com/YOUR_USERNAME/aitxhackathon-1-1/main/download.jpg
   ```

3. Update `hardcoded-data.ts`:
   ```typescript
   export const HARDCODED_FLOORPLAN: FloorPlan = {
     referenceImage: 'https://raw.githubusercontent.com/YOUR_USERNAME/aitxhackathon-1-1/main/download.jpg',
   }
   ```

4. Commit and push

5. **Done!** Everyone sees your floor plan automatically!

## ✅ What's Working Now

- ✅ App starts without errors
- ✅ 4 cameras are pre-placed
- ✅ Dashboard loads correctly
- ✅ No database required
- ✅ Everything hardcoded in Git

## 🚀 Next Steps

1. Choose one of the 3 options above
2. Add your floor plan
3. Test it works
4. Commit to GitHub
5. Teammates see everything!

---

**Status**: Error fixed, app running, 4 cameras pre-placed! 🎉

