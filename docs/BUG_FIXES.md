# Bug Fixes - December 13, 2025

## Bug #1: Missing MongoDB Fallback in Profile API ✅ FIXED

### **Problem:**
The `/api/auth/me` endpoint (both GET and PUT) lacked MongoDB error handling and fallback to in-memory storage. When MongoDB was unavailable, profile operations failed with "Internal server error" instead of gracefully falling back like the login and signup endpoints.

### **Impact:**
- Users could sign up and log in via fallback storage
- But profile updates and fetches would fail completely
- Inconsistent behavior across authentication endpoints

### **Root Cause:**
The endpoint directly called `prisma.user.findUnique()` and `prisma.user.update()` without try-catch blocks for MongoDB failures.

### **Solution:**
Updated `frontend/app/api/auth/me/route.ts`:

**GET Endpoint:**
- Added try-catch around Prisma calls
- Falls back to `inMemoryDB.findUserById()` if MongoDB fails
- Returns `databaseType` in response for transparency
- Logs which database was used

**PUT Endpoint:**
- Added try-catch around Prisma update
- Falls back to `inMemoryDB.updateUser()` if MongoDB fails
- Returns `databaseType` in response
- Logs which database was used

### **Code Changes:**
```typescript
// Before: Direct Prisma call (no fallback)
const user = await prisma.user.findUnique({
  where: { id: decoded.userId },
})

// After: MongoDB with fallback
let user
let dbType = 'In-Memory'

try {
  user = await prisma.user.findUnique({
    where: { id: decoded.userId },
  })
  if (user) {
    dbType = 'MongoDB'
    console.log('✅ User profile loaded from MongoDB')
  }
} catch (mongoError) {
  console.warn('⚠️ MongoDB unavailable, checking in-memory storage')
}

if (!user) {
  user = await inMemoryDB.findUserById(decoded.userId)
}
```

### **Testing:**
1. **With MongoDB running:** Profile updates save to MongoDB
2. **Without MongoDB:** Profile updates save to in-memory storage
3. **Consistency:** Same behavior as login/signup endpoints

---

## Bug #2: LivePage Loading from Deprecated localStorage ✅ FIXED

### **Problem:**
The `LivePage` component loaded cameras from the deprecated localStorage key `factory-floorplan-vectors`, which is no longer written to. The `DigitalTwinManual` component now saves exclusively to the API via `/api/user/data`.

### **Impact:**
- Cameras added in Camera Layout page were invisible on Live Cameras page
- Data was persisted on the server but not displayed
- Confusing user experience (cameras "disappeared")

### **Root Cause:**
`LivePage` had this code:
```typescript
const stored = localStorage.getItem('factory-floorplan-vectors')
```

But `DigitalTwinManual` removed all localStorage writes and now uses:
```typescript
await fetch('/api/user/data', {
  method: 'POST',
  body: JSON.stringify({ cameras, walls, referenceImage })
})
```

### **Solution:**
Updated `frontend/app/dashboard/live/page.tsx`:

**Before:**
- Loaded from localStorage
- Used deprecated key `factory-floorplan-vectors`
- No API integration

**After:**
- Loads from `/api/user/data` API endpoint
- Uses JWT authentication
- Converts API format to component format
- Logs success with database type
- Handles missing token gracefully

### **Code Changes:**
```typescript
// Before: localStorage (deprecated)
useEffect(() => {
  const stored = localStorage.getItem('factory-floorplan-vectors')
  if (stored) {
    const data = JSON.parse(stored)
    setCameras(data.cameras)
  }
}, [])

// After: API with authentication
useEffect(() => {
  const loadCameras = async () => {
    const token = localStorage.getItem('token')
    if (!token) return

    const response = await fetch('/api/user/data', {
      headers: { 'Authorization': `Bearer ${token}` }
    })

    const data = await response.json()
    if (data.cameras) {
      const loadedCameras = data.cameras.map(cam => ({
        id: cam.id,
        label: cam.label,
        streamUrl: cam.streamUrl,
        floor: cam.floor,
        active: cam.active !== false,
        status: cam.status || 'normal',
      }))
      setCameras(loadedCameras)
      console.log(`✅ Loaded ${loadedCameras.length} cameras`)
    }
  }
  loadCameras()
}, [])
```

### **Data Flow (Fixed):**
1. User places camera in Camera Layout → Saved to API
2. User navigates to Live Cameras → Loads from API
3. ✅ Cameras are visible and synced

### **Testing:**
1. Add cameras in Camera Layout page
2. Navigate to Live Cameras page
3. ✅ Cameras should appear immediately
4. Refresh page
5. ✅ Cameras should persist (loaded from MongoDB/in-memory)

---

## Summary

### **Files Modified:**
1. `frontend/app/api/auth/me/route.ts` - Added MongoDB fallback (NEW FILE)
2. `frontend/app/dashboard/live/page.tsx` - Changed from localStorage to API

### **Benefits:**
✅ **Consistency:** All auth endpoints now have MongoDB fallback  
✅ **Reliability:** Profile updates work even when MongoDB is down  
✅ **Data Sync:** Live Cameras page shows cameras from Camera Layout  
✅ **Persistence:** Cameras persist across page refreshes  
✅ **Transparency:** Logs show which database is being used  

### **Breaking Changes:**
None - these are pure bug fixes with no API changes.

### **Migration Notes:**
No migration needed. Existing users will automatically use the new API-based camera loading on their next visit to the Live Cameras page.

---

## Verification Checklist

- [x] Bug #1: Profile GET works with MongoDB fallback
- [x] Bug #1: Profile PUT works with MongoDB fallback
- [x] Bug #1: Console logs show database type
- [x] Bug #2: Live page loads cameras from API
- [x] Bug #2: Cameras persist across page refreshes
- [x] Bug #2: No localStorage dependency
- [x] No linter errors
- [x] Consistent with other auth endpoints

