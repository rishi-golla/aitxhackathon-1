# Troubleshooting Guide - Account Creation Issues

## Quick Diagnostics

### 1. Test API Connection
Open your browser and navigate to:
```
http://localhost:3000/api/test
```

You should see:
```json
{
  "status": "ok",
  "message": "API is working",
  "database": "connected",
  "userCount": 0
}
```

### 2. Check Browser Console
1. Open browser DevTools (F12)
2. Go to Console tab
3. Try to sign up
4. Look for error messages

Common errors and solutions:

#### Error: "Failed to fetch" or "Network error"
**Solution:** Make sure the dev server is running
```bash
cd frontend
npm run dev
```

#### Error: "Internal server error"
**Solution:** Check terminal for detailed error logs

#### Error: "Missing required fields"
**Solution:** Fill in all form fields (name, email, company, password)

### 3. Check Terminal Logs
Look at your terminal where `npm run dev` is running. You should see:
- No red error messages
- API requests being logged
- Any Prisma errors

### 4. Verify Database
Check if database exists and has correct schema:
```bash
cd frontend
npx prisma studio
```

This opens a web interface at `http://localhost:5555` where you can:
- View the User table
- Check if it has the correct columns
- Manually add a test user

### 5. Reset Database (if needed)
If the database is corrupted:
```bash
cd frontend
# Backup first (optional)
cp dev.db dev.db.backup

# Reset database
rm dev.db
npx prisma migrate dev --name init
npx prisma generate
```

## Common Issues and Solutions

### Issue 1: "Cannot find module '@/lib/prisma'"
**Solution:**
```bash
cd frontend
npx prisma generate
npm run dev
```

### Issue 2: Database locked
**Solution:**
1. Close Prisma Studio if open
2. Restart the dev server
3. Try again

### Issue 3: Port already in use
**Solution:**
```bash
# Kill process on port 3000
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Or use different port
npm run dev -- -p 3001
```

### Issue 4: JWT_SECRET not found
**Solution:**
Check `.env` file exists in `frontend/` directory:
```env
DATABASE_URL="file:./dev.db"
JWT_SECRET="your-secret-key-change-in-production"
```

## Manual Account Creation (Fallback)

If API still doesn't work, you can create an account directly in the database:

1. Open Prisma Studio:
```bash
cd frontend
npx prisma studio
```

2. Go to User table
3. Click "Add record"
4. Fill in:
   - email: your@email.com
   - password: (use bcrypt hash - see below)
   - fullName: Your Name
   - company: Your Company
   - role: operator
   - avatar: https://api.dicebear.com/7.x/initials/svg?seed=YourName

### Generate Password Hash
Use this Node.js command:
```bash
node -e "const bcrypt = require('bcryptjs'); console.log(bcrypt.hashSync('yourpassword', 10))"
```

## Still Not Working?

### Check these files exist:
- `frontend/dev.db` - Database file
- `frontend/.env` - Environment variables
- `frontend/lib/prisma.ts` - Prisma client
- `frontend/app/api/auth/signup/route.ts` - Signup API

### Verify Node modules:
```bash
cd frontend
npm list @prisma/client bcryptjs jsonwebtoken
```

Should show all three packages installed.

### Complete Reset:
```bash
cd frontend

# Remove node_modules and reinstall
rm -rf node_modules
rm package-lock.json
npm install --legacy-peer-deps

# Reset database
rm dev.db
npx prisma migrate dev --name init
npx prisma generate

# Restart server
npm run dev
```

## Getting More Help

1. Check terminal output for errors
2. Check browser console for errors
3. Check Network tab in DevTools
4. Look at the request/response for `/api/auth/signup`

### Expected Signup Request:
```json
POST /api/auth/signup
{
  "email": "user@example.com",
  "password": "password123",
  "fullName": "John Doe",
  "company": "Acme Corp",
  "role": "operator"
}
```

### Expected Response (Success):
```json
{
  "user": {
    "id": "...",
    "email": "user@example.com",
    "fullName": "John Doe",
    "company": "Acme Corp",
    "role": "operator",
    "avatar": "...",
    "createdAt": "...",
    "updatedAt": "..."
  },
  "token": "eyJ..."
}
```

### Expected Response (Error):
```json
{
  "error": "Email already registered"
}
```

