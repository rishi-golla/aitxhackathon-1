# Digital Twin Color Semantics

## Quick Reference Guide

### Camera Status Colors

```
🔵 TEAL (#3ddbd9)    → Normal Operation
   - Steady glow
   - Standard brightness
   - Everything is fine

🟡 AMBER (#f59e0b)   → Warning
   - Needs attention
   - May require action soon
   - Vertical beam indicator

🔴 RED (#ef4444)     → Incident
   - Active problem
   - Requires immediate action
   - Pulsing glow + beam

⚪ GRAY (#6b7280)    → Inactive
   - Offline or disabled
   - Muted appearance
   - No glow effects
```

## Visual Indicators

### Camera Dot Size
- **Normal**: 0.09 radius
- **Hover**: 0.14 radius (1.6× scale)
- **Selected**: 0.14 radius + ring

### Glow Intensity
- **Normal**: 3.0
- **Warning**: 3.0
- **Incident**: 4.0 (pulsing 0.4-1.2)
- **Inactive**: 1.5

### Additional Effects

**Vertical Beam** (Warning/Incident only)
- Height: 1 unit
- Radius: 0.02
- Opacity: 30%
- Color: Matches camera state

**Pulsing Ring** (Hover/Select)
- Inner radius: 0.16
- Outer radius: 0.22
- Opacity: 50%
- Rotation: Animated

**Point Light Glow**
- Normal: 0.5 intensity
- Warning: 0.5 intensity
- Incident: 0.8 intensity (pulsing)
- Distance: 2 units

## Color Distribution (Demo)

In the current demo, cameras are randomly assigned states:

- **75%** Normal (Teal) - Most cameras operating fine
- **15%** Warning (Amber) - Some need attention
- **10%** Incident (Red) - Few critical issues
- **Variable** Inactive (Gray) - Based on `camera.active` flag

## Reading the Scene

### At a Glance
1. **Scan for red** - Immediate problems
2. **Check amber** - Upcoming issues
3. **Teal is baseline** - Normal operation
4. **Gray is expected** - Known offline cameras

### Depth Perception
- **Closer objects**: Brighter, more saturated
- **Farther objects**: Faded, less prominent
- **Fog effect**: Natural distance cues

### Visual Hierarchy
```
Priority 1: Red pulsing cameras (incidents)
Priority 2: Amber cameras with beams (warnings)
Priority 3: Teal cameras (normal)
Priority 4: Walls and structure
Priority 5: Grid (reference only)
Priority 6: Floor planes
```

## Integration with Real Systems

To connect to real camera data, update `getCameraState()` in `CameraNodeRenderer.tsx`:

```typescript
const getCameraState = (camera: CameraNode) => {
  // Replace demo logic with real API data
  if (!camera.active) {
    return { color: '#6b7280', label: 'inactive', intensity: 1.5 }
  }
  
  // Check camera.status from your API
  switch (camera.status) {
    case 'incident':
      return { color: '#ef4444', label: 'incident', intensity: 4 }
    case 'warning':
      return { color: '#f59e0b', label: 'warning', intensity: 3 }
    default:
      return { color: '#3ddbd9', label: 'normal', intensity: 3 }
  }
}
```

## Accessibility Notes

- Colors chosen for common color blindness types
- Red/Teal have high contrast
- Amber is distinct from both
- Gray is clearly different
- Additional indicators (beams, pulsing) provide non-color cues

## Legend Location

Bottom-left corner of 3D view:
```
● Normal  ● Warning  ● Incident  ● Inactive
```

Always visible, semi-transparent background, no interaction required.

