# Quick Start Guide

## ✅ The App is Now Running!

I've fixed the issue and the app now works **without requiring MongoDB**!

### What I Fixed:
1. ✅ Cleared Next.js cache
2. ✅ Added in-memory database fallback
3. ✅ Restarted the dev server
4. ✅ API routes now work properly

## 🚀 Try It Now!

### Step 1: Open the App
```
http://localhost:3000
```

### Step 2: Test the API
```
http://localhost:3000/api/test
```

You should see:
```json
{
  "status": "ok",
  "message": "API is working",
  "database": "connected",
  "databaseType": "In-Memory (Fallback)",
  "userCount": 0,
  "note": "MongoDB not available. Using temporary in-memory storage..."
}
```

### Step 3: Create an Account
1. Click "Sign up"
2. Fill in the form:
   - Full Name: John Doe
   - Email: john@example.com
   - Company: Acme Corp
   - Role: Operator
   - Password: password123
   - Confirm Password: password123
3. Click "Create account"
4. You'll be logged in automatically! 🎉

## 📝 Important Notes

### Using In-Memory Storage
Since MongoDB isn't installed, the app uses **temporary in-memory storage**:

✅ **Pros:**
- Works immediately, no setup needed
- Fast and simple
- Perfect for development/testing

⚠️ **Cons:**
- Data is lost when you restart the server
- Not suitable for production
- Can't share data between multiple instances

### Want to Use MongoDB Instead?

If you want persistent data storage:

1. **Install MongoDB:**
   - Download from: https://www.mongodb.com/try/download/community
   - Install and start the service

2. **Restart the app:**
   ```bash
   cd frontend
   npm run dev
   ```

3. **The app will automatically detect MongoDB and use it!**

## 🎯 What You Can Do Now

### 1. Create Multiple Accounts
- Each account is stored in memory
- You can login/logout
- Profile customization works

### 2. Explore the Dashboard
- View the camera layout
- See accident reports
- Check live feeds
- Customize your profile

### 3. Test All Features
- Upload floor plans
- Place cameras
- View 3D layout
- Check recent activity

## 🔄 Restarting the App

Every time you restart the dev server, you'll need to create a new account (since data is in memory).

**To restart:**
1. Press `Ctrl+C` in the terminal
2. Run: `npm run dev`
3. Create a new account

## 💾 Persistent Storage (Optional)

Want to keep your data? Install MongoDB:

### Windows:
1. Download: https://www.mongodb.com/try/download/community
2. Run installer
3. Check "Install as Windows Service"
4. Restart your app - it will automatically use MongoDB!

### Mac:
```bash
brew install mongodb-community
brew services start mongodb-community
```

### Linux:
```bash
sudo apt-get install mongodb
sudo systemctl start mongod
```

Then just restart the app - no configuration needed!

## 🐛 Troubleshooting

### "Server error: Expected JSON response but got HTML"
**Solution:** The dev server needs to be restarted.
```bash
cd frontend
npm run dev
```

### Can't create an account
**Check:**
1. Dev server is running: `http://localhost:3000`
2. API works: `http://localhost:3000/api/test`
3. Browser console for errors (F12)

### Port 3000 already in use
```bash
# Kill the process
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Or use different port
npm run dev -- -p 3001
```

## 📚 Next Steps

1. **Explore the Dashboard** - Check out all the features
2. **Upload a Floor Plan** - Test the 3D camera placement
3. **Customize Your Profile** - Try different avatar styles
4. **View Accident Reports** - See the mock data

## 🎉 You're All Set!

The app is working and ready to use. No MongoDB required for now!

If you want persistent storage later, just install MongoDB and restart - it will automatically switch from in-memory to MongoDB.

Enjoy! 🚀

