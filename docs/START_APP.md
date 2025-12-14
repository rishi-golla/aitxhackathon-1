# How to Start the Application

## ⚠️ IMPORTANT: Follow these steps in order

### Step 1: Start MongoDB

**Check if MongoDB is running:**
```bash
mongosh mongodb://localhost:27017
```

If you get an error, start MongoDB:

**Windows:**
```bash
net start MongoDB
```

**Mac:**
```bash
brew services start mongodb-community
```

**Linux:**
```bash
sudo systemctl start mongod
```

### Step 2: Verify Database Setup

```bash
cd frontend
npx prisma generate
npx prisma db push
```

You should see:
```
✔ Generated Prisma Client
Your database indexes are now in sync with your Prisma schema.
```

### Step 3: Start the Development Server

```bash
cd frontend
npm run dev
```

Wait for:
```
✓ Ready in X.Xs
○ Local: http://localhost:3000
```

### Step 4: Open the Application

Open your browser and go to:
```
http://localhost:3000
```

You should be redirected to the login page.

## 🧪 Test the API

Before creating an account, test if the API is working:

1. Open: http://localhost:3000/api/test
2. You should see JSON like:
```json
{
  "status": "ok",
  "message": "API is working",
  "database": "connected",
  "userCount": 0
}
```

If you see HTML or an error, the server isn't running correctly.

## 🐛 Troubleshooting

### Error: "Unexpected token '<', "<!DOCTYPE "... is not valid JSON"

**This means the API routes aren't working.**

**Solution:**
1. **Stop the dev server** (Ctrl+C)
2. **Clear Next.js cache:**
   ```bash
   cd frontend
   Remove-Item -Recurse -Force .next
   ```
3. **Restart:**
   ```bash
   npm run dev
   ```

### Error: "MongoServerError: connect ECONNREFUSED"

**MongoDB is not running.**

**Solution:**
```bash
# Windows
net start MongoDB

# Mac
brew services start mongodb-community

# Linux
sudo systemctl start mongod
```

### Error: "Cannot find module '@/lib/prisma'"

**Prisma Client not generated.**

**Solution:**
```bash
cd frontend
npx prisma generate
```

### Port 3000 already in use

**Solution:**
```bash
# Find and kill the process
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Or use a different port
npm run dev -- -p 3001
```

## 📝 Creating Your First Account

1. Go to http://localhost:3000
2. Click "Sign up"
3. Fill in:
   - Full Name: Your Name
   - Email: your@email.com
   - Company: Your Company
   - Role: Select one
   - Password: At least 6 characters
   - Confirm Password: Same as above
4. Click "Create account"
5. You'll be redirected to the dashboard

## ✅ Verify Everything Works

### Check Database
```bash
mongosh nda-company
db.User.find().pretty()
```

You should see your user account.

### Check Prisma Studio
```bash
cd frontend
npx prisma studio
```

Opens at http://localhost:5555 - you can view/edit data here.

## 🔄 Daily Startup Routine

Every time you want to work on the project:

1. **Start MongoDB** (if not running)
   ```bash
   net start MongoDB
   ```

2. **Start Dev Server**
   ```bash
   cd frontend
   npm run dev
   ```

3. **Open Browser**
   ```
   http://localhost:3000
   ```

That's it! 🎉

## 🛑 Stopping the Application

1. **Stop Dev Server:** Press `Ctrl+C` in terminal
2. **Stop MongoDB (optional):**
   ```bash
   # Windows
   net stop MongoDB
   
   # Mac
   brew services stop mongodb-community
   
   # Linux
   sudo systemctl stop mongod
   ```

## 📚 Additional Resources

- **MongoDB Setup:** See `MONGODB_SETUP.md`
- **Database Info:** See `DATABASE_SETUP.md`
- **Troubleshooting:** See `TROUBLESHOOTING.md`

