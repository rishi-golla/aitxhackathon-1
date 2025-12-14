# UI Controls Reference

## Button Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  [Upload Floor Plan]  [Floor: 1 2 3 4 5]                       │
│                                                                  │
│  [Auto-Trace] [Manual Trace] [Place Cameras] [Clear Walls]     │
│                                                                  │
│  [120 Walls] [5 Cameras]                                        │
└─────────────────────────────────────────────────────────────────┘
```

## Button States

### Upload Floor Plan
- **Default**: Gray button, white text
- **After Upload**: Shows "Change Reference"
- **Disabled**: Never (always clickable)

### Auto-Trace (Gradient Cyan/Blue)
- **Default**: Gradient button with checkmark icon
- **Disabled**: No image uploaded OR currently tracing
- **Active**: Shows spinner + "Auto-Tracing..."
- **After**: Walls appear in 3D view

### Manual Trace
- **Default**: Gray button, white text
- **Active**: Cyan button, "✓ Manual Trace"
- **Disabled**: No image uploaded OR auto-tracing
- **Exit**: Click again to return to view mode

### Place Cameras
- **Default**: Gray button, white text
- **Active**: Cyan button, "✓ Placing Cameras"
- **Disabled**: Auto-tracing in progress
- **Exit**: Click again to return to view mode

### Clear Walls
- **Default**: Gray button, red text/border
- **Visible**: Only when walls exist
- **Action**: Confirms before deleting all walls
- **Disabled**: Auto-tracing in progress

## Floor Selector

```
Floor: [1] [2] [3] [4] [5]
       ^^^
     Active (cyan glow)
```

- **Active Floor**: Cyan background, dark text, glow effect
- **Inactive**: Gray text, transparent background
- **Hover**: White text, subtle background
- **Function**: Shows only walls/cameras on selected floor

## Status Indicators

```
[●] 120 Walls    [●] 5 Cameras
 ^                ^
Cyan dot        Cyan dot
```

- Updates in real-time
- Shows count for current floor only
- Cyan dot = active/live status

## Mode Indicators

### View Mode (Default)
- All buttons gray/inactive
- Can orbit, zoom, pan
- Click cameras to view details

### Trace Walls Mode
- "Manual Trace" button is cyan
- Reference image visible (20% opacity)
- Floor shows cyan preview line
- Bottom hint: "Click to place wall points • Press Enter to finish • Esc to cancel"

### Place Cameras Mode
- "Place Cameras" button is cyan
- Click anywhere to place camera
- Opens modal for camera details
- Bottom hint: "Click anywhere on the floor to place a camera"

### Auto-Tracing Mode
- "Auto-Trace" button shows spinner
- All other buttons disabled
- Takes 2-5 seconds
- Shows alert when complete

## Keyboard Shortcuts

| Key | Action | Mode |
|-----|--------|------|
| **Enter** | Finish wall tracing | Trace Walls |
| **Esc** | Cancel tracing | Trace Walls |
| **Ctrl+Z** | Undo last point | Trace Walls |
| **Mouse Drag** | Orbit camera | All modes |
| **Scroll** | Zoom in/out | All modes |
| **Right Drag** | Pan view | All modes |

## Visual Feedback

### During Auto-Trace
```
[Auto-Trace]  →  [⟳ Auto-Tracing...]  →  Alert: "Auto-traced 120 walls!"
```

### During Manual Trace
```
Click 1: Cyan dot appears
Click 2: Cyan line + dot
Click 3: More lines + dot
Enter:   Lines become walls (gray with cyan edges)
```

### Camera Placement
```
Click floor  →  Modal opens  →  Enter details  →  Cyan dot appears
```

## Color Guide

- **Cyan (#3ddbd9)**: Active elements, walls, cameras
- **Blue (#00F5CC)**: Accent, hover states
- **Gray (#3a4350)**: Wall solids
- **Red (#EF4444)**: Delete/clear actions
- **White/Zinc**: Text, UI elements
- **Dark (#0a0c10)**: Background

## Responsive Behavior

- Buttons wrap on smaller screens
- Floor selector always visible
- Status indicators stack vertically if needed
- 3D view fills remaining space

