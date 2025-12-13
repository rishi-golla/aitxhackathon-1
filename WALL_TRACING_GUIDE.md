# Manual Wall Tracing System

## Overview
Professional CAD-style wall tracing system for factory digital twins. No automatic detection - all geometry is user-drawn.

## User Flow

### 1. Upload Floor Plan
- Click "Upload Floor Plan"
- Select PNG/JPG reference image
- Image displays as locked reference layer (20% opacity)

### 2. Trace Walls (Two Options)

#### Option A: Auto-Trace (Recommended)
- Click **"Auto-Trace"** button (gradient cyan/blue)
- System automatically detects walls from image
- Takes 2-5 seconds depending on image complexity
- Detects up to 150 wall segments
- Results appear immediately in 3D view
- Can manually adjust or add more walls after

#### Option B: Manual Trace
- Click **"Manual Trace"** button
- Click on floor to place wall points
- Points snap together (0.3 unit threshold)
- Press **Enter** to finish tracing
- Press **Esc** to cancel
- Press **Ctrl+Z** to undo last point

**Best Practice**: Use Auto-Trace first, then Manual Trace to refine or add missing walls.

### 3. Place Cameras
- Click "Place Cameras" button
- Click anywhere on floor to place camera
- Enter camera name and stream URL
- Camera appears as glowing cyan dot

### 4. View Mode
- Default mode for navigation
- Click camera dots to view details
- Change floors using floor selector

## Data Model

```typescript
type WallSegment = {
  id: string
  start: [number, number]  // 2D floor coordinates
  end: [number, number]
  floor: number
}

type CameraNode = {
  id: string
  position: [number, number, number]
  rotation: [number, number, number]
  floor: number
  label: string
  streamUrl: string
  active: boolean
}
```

## Storage
- All vectors saved to `localStorage`
- Key: `factory-floorplan-vectors`
- Persists across sessions
- Reference image stored as base64

## Rendering

### Walls
- Thin extruded strips (0.4 units height, 0.04 units thick)
- Matte gray material (#3a4350)
- Cyan edge highlights (#3ddbd9)
- Slight elevation (0.02 units above floor)

### Cameras
- Glowing cyan spheres (0.08 radius)
- Emissive material (intensity 2.5)
- Floating animation
- Pulsing ring on hover/select

## Controls

### Keyboard
- **Enter**: Complete wall tracing
- **Esc**: Cancel tracing
- **Ctrl+Z**: Undo last point

### Mouse
- **Click**: Place point / camera
- **Drag**: Orbit camera
- **Scroll**: Zoom
- **Right-drag**: Pan

## Auto-Trace Algorithm

### Detection Process
1. **Image Processing** (1200px resolution)
   - Convert to grayscale
   - Enhance contrast (normalize min/max)
   
2. **Edge Detection**
   - Sobel operator for gradient calculation
   - Adaptive threshold (40 units)
   - Detects thick walls (checks ±2 pixels)

3. **Line Detection**
   - Scans every row (horizontal lines)
   - Scans every column (vertical lines)
   - Minimum length: 4% of image dimension
   - Allows 20px gaps to bridge broken lines
   - Requires 25% edge coverage

4. **Line Merging**
   - Merges lines within 10px of each other
   - Allows 40px overlap/gap for connection
   - Sorts by strength (edge count)
   - Keeps top 150 strongest lines

5. **Coordinate Conversion**
   - Maps image pixels to 3D world space
   - Centers on origin (0, 0)
   - Scales to 14×14 unit floor

### When Auto-Trace Works Best
- ✅ High contrast architectural drawings
- ✅ Clear black/dark walls on white/light background
- ✅ Straight walls (horizontal/vertical)
- ✅ Clean scanned floor plans
- ✅ CAD exports

### When to Use Manual Trace
- ❌ Low contrast images
- ❌ Photos of physical drawings
- ❌ Curved or angled walls
- ❌ Heavily annotated plans
- ❌ Auto-trace missed walls

## Components

- `DigitalTwinManual.tsx` - Main orchestrator with auto-trace
- `FactorySceneManual.tsx` - 3D scene setup
- `WallTraceMode.tsx` - Interactive wall drawing
- `WallRenderer.tsx` - Renders saved walls
- `CameraNodeRenderer.tsx` - Renders camera dots
- `ReferenceImagePlane.tsx` - Floor plan reference
- `autoTrace.ts` - Automatic wall detection algorithm

## Validation

✅ No automatic geometry generation
✅ No room blocks or cuboids
✅ Walls are lines, not volumes
✅ User must trace every wall
✅ Reference image is visual only
✅ All geometry from user vectors

