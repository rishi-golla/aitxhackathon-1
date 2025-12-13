# Auto-Trace vs Manual Trace Comparison

## Quick Comparison

| Feature | Auto-Trace | Manual Trace |
|---------|-----------|--------------|
| **Speed** | 2-5 seconds | 2-10 minutes |
| **Accuracy** | 70-90% | 100% |
| **Best For** | Initial layout | Refinement |
| **Skill Required** | None | Basic CAD knowledge |
| **Works With** | Clean drawings | Any image |
| **Max Walls** | 150 segments | Unlimited |

## Recommended Workflow

### Step 1: Auto-Trace First
```
Upload Floor Plan → Click "Auto-Trace" → Wait 2-5 sec → Review Results
```

**Expected Results:**
- ✅ Major walls detected (exterior, main rooms)
- ✅ Grid-aligned walls captured
- ⚠️ Some details may be missed
- ⚠️ May have extra noise lines

### Step 2: Manual Refinement
```
Click "Manual Trace" → Add missing walls → Press Enter
```

**What to Add:**
- Doorways and openings
- Curved or angled walls
- Small interior partitions
- Walls in low-contrast areas

### Step 3: Clean Up (Optional)
```
Click "Clear Walls" → Start fresh if needed
```

## Auto-Trace Performance by Image Type

### ⭐⭐⭐⭐⭐ Excellent (90%+ accuracy)
- CAD exports (DWG → PNG)
- Vector floor plans
- High-contrast architectural drawings
- Black lines on white background

### ⭐⭐⭐⭐ Good (70-90% accuracy)
- Scanned architectural plans
- Blueprint PDFs
- Clean hand drawings
- Most professional floor plans

### ⭐⭐⭐ Fair (50-70% accuracy)
- Photos of drawings
- Low-resolution images
- Colored floor plans
- Annotated drawings

### ⭐⭐ Poor (<50% accuracy)
- Hand-sketches
- Perspective photos
- Heavily textured images
- Very low contrast

## Tips for Best Results

### Before Upload
1. **Crop** - Remove borders, legends, and annotations
2. **Rotate** - Ensure walls are horizontal/vertical
3. **Enhance** - Increase contrast if possible
4. **Resolution** - Use at least 800×800px

### After Auto-Trace
1. **Zoom In** - Check wall accuracy
2. **Count Walls** - Should match major walls in image
3. **Look for Noise** - Delete stray lines if needed
4. **Add Details** - Use manual trace for missing walls

## Hybrid Approach Example

```
1. Upload clean floor plan
2. Auto-Trace → 120 walls detected
3. Review: 95% accurate, missing 3 interior walls
4. Manual Trace → Add 3 missing walls
5. Total: 123 walls in ~30 seconds
```

**Time Saved:** 95% faster than full manual trace!

## Troubleshooting

### "No walls detected"
- Image contrast too low
- Walls too thin/light
- Try manual trace instead

### "Too many walls detected"
- Image has text/annotations
- Crop image and re-upload
- Use "Clear Walls" and manual trace

### "Walls in wrong places"
- Image not aligned properly
- Rotate image before upload
- Use manual trace for precision

### "Auto-Trace button disabled"
- Upload floor plan first
- Exit manual trace mode
- Wait for previous trace to complete

## When to Skip Auto-Trace

Use **Manual Trace Only** if:
- ❌ Image is a photo (not scan/export)
- ❌ Walls are curved or at odd angles
- ❌ Floor plan is hand-drawn sketch
- ❌ You need pixel-perfect accuracy
- ❌ Image has heavy annotations

## Performance Notes

- **Processing Time**: 2-5 seconds for 1200×1200px image
- **Max Detection**: 150 wall segments (strongest lines)
- **Memory Usage**: ~10MB during processing
- **Browser Support**: All modern browsers (uses Canvas API)

