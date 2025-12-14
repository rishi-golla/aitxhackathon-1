# MongoDB Setup Guide

## Quick Start

### 1. Check if MongoDB is Running
```bash
mongosh mongodb://localhost:27017
```

If you see a connection, MongoDB is running! ✅

If you get an error, follow the installation steps below.

## Installation

### Windows

1. **Download MongoDB Community Server**
   - Visit: https://www.mongodb.com/try/download/community
   - Select: Windows, MSI installer
   - Download and run the installer

2. **Install as Windows Service**
   - During installation, check "Install MongoDB as a Service"
   - Use default settings
   - Service will start automatically

3. **Verify Installation**
   ```bash
   mongosh
   ```

### macOS

```bash
# Install using Homebrew
brew tap mongodb/brew
brew install mongodb-community

# Start MongoDB
brew services start mongodb-community

# Verify
mongosh
```

### Linux (Ubuntu/Debian)

```bash
# Import MongoDB public key
wget -qO - https://www.mongodb.org/static/pgp/server-7.0.asc | sudo apt-key add -

# Add MongoDB repository
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list

# Update and install
sudo apt-get update
sudo apt-get install -y mongodb-org

# Start MongoDB
sudo systemctl start mongod
sudo systemctl enable mongod

# Verify
mongosh
```

## Start/Stop MongoDB

### Windows
```bash
# Start
net start MongoDB

# Stop
net stop MongoDB

# Check status
sc query MongoDB
```

### macOS
```bash
# Start
brew services start mongodb-community

# Stop
brew services stop mongodb-community

# Restart
brew services restart mongodb-community
```

### Linux
```bash
# Start
sudo systemctl start mongod

# Stop
sudo systemctl stop mongod

# Restart
sudo systemctl restart mongod

# Status
sudo systemctl status mongod
```

## Using MongoDB with the Application

### 1. Ensure MongoDB is Running
```bash
mongosh mongodb://localhost:27017
```

### 2. Push Database Schema
```bash
cd frontend
npx prisma db push
```

### 3. Start the Application
```bash
cd frontend
npm run dev
```

### 4. Create Your First Account
- Open: http://localhost:3000
- Click "Sign up"
- Fill in the form
- Your data is now stored in MongoDB!

## Viewing Your Data

### Option 1: Prisma Studio (Recommended)
```bash
cd frontend
npx prisma studio
```
Opens a web interface at http://localhost:5555

### Option 2: MongoDB Shell
```bash
mongosh nda-company

# View all users
db.User.find().pretty()

# View all cameras
db.Camera.find().pretty()

# View all accidents
db.Accident.find().pretty()

# Count documents
db.User.countDocuments()
```

### Option 3: MongoDB Compass (GUI)
1. Download: https://www.mongodb.com/try/download/compass
2. Connect to: `mongodb://localhost:27017`
3. Browse the `nda-company` database

## Common Commands

### Create Database Backup
```bash
mongodump --db=nda-company --out=./backup
```

### Restore Database
```bash
mongorestore --db=nda-company ./backup/nda-company
```

### Drop Database (Reset)
```bash
mongosh nda-company --eval "db.dropDatabase()"
```

### View All Databases
```bash
mongosh --eval "show dbs"
```

## Troubleshooting

### Error: "MongoServerError: connect ECONNREFUSED"
**Solution:** MongoDB is not running. Start it:
```bash
# Windows
net start MongoDB

# Mac
brew services start mongodb-community

# Linux
sudo systemctl start mongod
```

### Error: "Data directory not found"
**Solution:** Create the data directory:
```bash
# Windows
mkdir C:\data\db

# Mac/Linux
sudo mkdir -p /data/db
sudo chown -R `id -un` /data/db
```

### Error: "Port 27017 already in use"
**Solution:** Another MongoDB instance is running
```bash
# Find the process
netstat -ano | findstr :27017

# Kill it (Windows)
taskkill /PID <PID> /F

# Or use a different port
mongod --port 27018
```

### Can't connect from application
**Check:**
1. MongoDB is running: `mongosh`
2. .env file has correct URL: `DATABASE_URL="mongodb://localhost:27017/nda-company"`
3. Restart dev server: `npm run dev`

## Configuration

### Change Database Name
Edit `frontend/.env`:
```env
DATABASE_URL="mongodb://localhost:27017/your-database-name"
```

Then:
```bash
cd frontend
npx prisma db push
```

### Use MongoDB Atlas (Cloud)
1. Create account at https://www.mongodb.com/cloud/atlas
2. Create a free cluster
3. Get connection string
4. Update `.env`:
```env
DATABASE_URL="mongodb+srv://username:password@cluster.mongodb.net/nda-company"
```

## Security (Production)

### Enable Authentication
```bash
# Create admin user
mongosh admin
db.createUser({
  user: "admin",
  pwd: "strongpassword",
  roles: ["root"]
})

# Update connection string
DATABASE_URL="mongodb://admin:strongpassword@localhost:27017/nda-company?authSource=admin"
```

### Enable SSL/TLS
```bash
mongod --tlsMode requireTLS --tlsCertificateKeyFile /path/to/cert.pem
```

## Resources

- MongoDB Docs: https://docs.mongodb.com/
- Prisma MongoDB Guide: https://www.prisma.io/docs/concepts/database-connectors/mongodb
- MongoDB University (Free): https://university.mongodb.com/

