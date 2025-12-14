# Camera Positioning Guide

## ✅ Floor Plan Now Loading!

Your `download.jpg` (321x157 pixels) is now set as the default floor plan and will load automatically.

## 📸 Current Camera Setup

4 cameras are now positioned at the corners of your layout:

| Camera | Label | Position | Location |
|--------|-------|----------|----------|
| cam-1 | Camera 1 - Top Left | `[-4, 0, 2]` | Top left corner |
| cam-2 | Camera 2 - Top Right | `[4, 0, 2]` | Top right corner |
| cam-3 | Camera 3 - Bottom Left | `[-4, 0, -2]` | Bottom left corner |
| cam-4 | Camera 4 - Bottom Right | `[4, 0, -2]` | Bottom right corner |

## 🎯 How 3D Positions Work

The position format is `[x, y, z]`:
- **X**: Left (-) to Right (+)
- **Y**: Height (usually 0 for floor level)
- **Z**: Back (-) to Front (+)

Your floor plan is centered at `[0, 0, 0]`.

## 🔧 Adjusting Camera Positions

### Method 1: Place Cameras in the App (Easiest)

1. **Run the app** (it's running now!)
2. **Click on your floor plan** where you want cameras
3. **Cameras will be placed** at those positions
4. **Export the data**:
   ```powershell
   Invoke-WebRequest -Uri "http://localhost:3000/api/user/data" | Select-Object -ExpandProperty Content > camera-positions.json
   ```
5. **Open `camera-positions.json`** and copy the camera positions
6. **Update `frontend/lib/hardcoded-data.ts`** with the new positions
7. **Commit to GitHub**

### Method 2: Edit Positions Manually

Edit `frontend/lib/hardcoded-data.ts`:

```typescript
export const HARDCODED_CAMERAS: Camera[] = [
  {
    id: 'cam-1',
    label: 'Entrance Camera',
    streamUrl: 'rtsp://demo.com/stream1',
    floor: 1,
    position: JSON.stringify([2, 0, 3]), // CHANGE THESE NUMBERS
    rotation: JSON.stringify([0, 0, 0]),
    status: 'normal',
    active: true,
  },
  // ... more cameras
]
```

**Tips for positioning**:
- Start with small numbers (between -5 and 5)
- Increase X to move right, decrease to move left
- Increase Z to move forward, decrease to move back
- Y is usually 0 (floor level)

## 📐 Example Positions for Different Layouts

### Corners (Current Setup):
```typescript
Top Left:     [-4, 0,  2]
Top Right:    [ 4, 0,  2]
Bottom Left:  [-4, 0, -2]
Bottom Right: [ 4, 0, -2]
```

### Along Walls:
```typescript
Left Wall:    [-5, 0,  0]
Right Wall:   [ 5, 0,  0]
Top Wall:     [ 0, 0,  3]
Bottom Wall:  [ 0, 0, -3]
```

### Center Areas:
```typescript
Center:       [ 0, 0,  0]
Center-Left:  [-2, 0,  0]
Center-Right: [ 2, 0,  0]
```

## 🎨 Camera Status Colors

Change the `status` field to change camera colors:
- `'normal'` - Teal/cyan (everything OK)
- `'warning'` - Amber/yellow (needs attention)
- `'incident'` - Red (alert!)
- `'inactive'` - Gray (offline)

```typescript
{
  id: 'cam-1',
  label: 'High Risk Area',
  // ...
  status: 'warning', // Makes it amber/yellow
  active: true,
}
```

## 🔄 Workflow

1. **Open app** → See current camera positions
2. **Not happy?** → Adjust in app or edit code
3. **Export positions** → Get exact coordinates
4. **Update hardcoded-data.ts** → Make permanent
5. **Commit** → Everyone sees your layout!

## 📝 Current Files

- **Floor Plan**: `frontend/public/download.jpg` (321x157px)
- **Camera Data**: `frontend/lib/hardcoded-data.ts`
- **4 Cameras**: Positioned at corners

## ✅ What Should Happen Now

When you open the app:
1. ✅ Your floor plan (`download.jpg`) loads
2. ✅ 4 cameras appear at the corners
3. ✅ You can click to add more cameras
4. ✅ Auto-trace walls from the image

## 🎯 Next Steps

1. **Check the app** - See if cameras are where you want them
2. **Adjust if needed** - Use Method 1 or 2 above
3. **Commit changes** - Make it permanent for everyone

---

**The app is now running with your floor plan and 4 corner cameras!** 🎉

