'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { motion } from 'framer-motion'

interface User {
  id: string
  fullName: string
  email: string
  company: string
  role: string
  createdAt: string
  avatar: string
}

const EditIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M11 2L14 5L5 14H2V11L11 2Z" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
)

const LogoutIcon = () => (
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M6 14H3C2.44772 14 2 13.5523 2 13V3C2 2.44772 2.44772 2 3 2H6" strokeLinecap="round" />
    <path d="M11 11L14 8L11 5" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M14 8H6" strokeLinecap="round" />
  </svg>
)

const avatarStyles = [
  { name: 'Initials', value: 'initials', preview: 'AB' },
  { name: 'Avataaars', value: 'avataaars', preview: '👤' },
  { name: 'Bottts', value: 'bottts', preview: '🤖' },
  { name: 'Identicon', value: 'identicon', preview: '🔷' },
  { name: 'Pixel Art', value: 'pixel-art', preview: '👾' },
  { name: 'Shapes', value: 'shapes', preview: '🔶' },
]

const themeColors = [
  { name: 'Cyan', value: 'cyan', color: '#28e7d3' },
  { name: 'Blue', value: 'blue', color: '#3b82f6' },
  { name: 'Purple', value: 'purple', color: '#a855f7' },
  { name: 'Pink', value: 'pink', color: '#ec4899' },
  { name: 'Green', value: 'green', color: '#10b981' },
  { name: 'Orange', value: 'orange', color: '#f59e0b' },
]

export default function ProfilePage() {
  const router = useRouter()
  const [user, setUser] = useState<User | null>(null)
  const [isEditing, setIsEditing] = useState(false)
  const [formData, setFormData] = useState({
    fullName: '',
    email: '',
    company: '',
    role: '',
  })
  const [selectedAvatarStyle, setSelectedAvatarStyle] = useState('initials')
  const [selectedTheme, setSelectedTheme] = useState('cyan')
  const [notifications, setNotifications] = useState({
    email: true,
    push: false,
    accidents: true,
    cameras: true,
  })

  useEffect(() => {
    // Load current user
    const currentUser = localStorage.getItem('currentUser')
    if (currentUser) {
      const userData = JSON.parse(currentUser)
      setUser(userData)
      setFormData({
        fullName: userData.fullName,
        email: userData.email,
        company: userData.company,
        role: userData.role,
      })
      
      // Load preferences
      const prefs = localStorage.getItem('userPreferences')
      if (prefs) {
        const preferences = JSON.parse(prefs)
        setSelectedAvatarStyle(preferences.avatarStyle || 'initials')
        setSelectedTheme(preferences.theme || 'cyan')
        setNotifications(preferences.notifications || notifications)
      }
    } else {
      // Redirect to login if no user
      router.push('/login')
    }
  }, [router])

  const handleSave = async () => {
    if (!user) return

    try {
      const token = localStorage.getItem('token')
      const response = await fetch('/api/auth/me', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify(formData),
      })

      const data = await response.json()

      if (response.ok) {
        setUser(data.user)
        localStorage.setItem('currentUser', JSON.stringify(data.user))
        setIsEditing(false)
      }
    } catch (error) {
      console.error('Failed to update profile:', error)
    }
  }

  const handleAvatarChange = (style: string) => {
    setSelectedAvatarStyle(style)
    savePreferences({ avatarStyle: style })
    
    // Update avatar URL
    if (user) {
      const newAvatar = `https://api.dicebear.com/7.x/${style}/svg?seed=${user.fullName}`
      const updatedUser = { ...user, avatar: newAvatar }
      setUser(updatedUser)
      localStorage.setItem('currentUser', JSON.stringify(updatedUser))
    }
  }

  const handleThemeChange = (theme: string) => {
    setSelectedTheme(theme)
    savePreferences({ theme })
    // You can implement theme switching logic here
  }

  const handleNotificationChange = (key: string, value: boolean) => {
    const updated = { ...notifications, [key]: value }
    setNotifications(updated)
    savePreferences({ notifications: updated })
  }

  const savePreferences = (prefs: any) => {
    const current = JSON.parse(localStorage.getItem('userPreferences') || '{}')
    const updated = { ...current, ...prefs }
    localStorage.setItem('userPreferences', JSON.stringify(updated))
  }

  const handleLogout = () => {
    localStorage.removeItem('currentUser')
    localStorage.removeItem('token')
    router.push('/login')
  }

  if (!user) {
    return (
      <div className="flex-1 flex items-center justify-center h-full">
        <div className="text-white/60">Loading...</div>
      </div>
    )
  }

  const accountAge = Math.floor(
    (new Date().getTime() - new Date(user.createdAt).getTime()) / (1000 * 60 * 60 * 24)
  )

  return (
    <div className="flex-1 p-6 overflow-auto h-full">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-semibold text-white mb-1">Profile</h1>
        <p className="text-white/50 text-sm">Manage your account settings and preferences</p>
      </div>

      <div className="grid grid-cols-[1fr_400px] gap-6">
        {/* Main Profile Card */}
        <div className="space-y-6">
          {/* Profile Info */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-2xl bg-card border border-white/5 p-6"
          >
            <div className="flex items-start justify-between mb-6">
              <h2 className="text-xl font-semibold text-white">Personal Information</h2>
              {!isEditing ? (
                <button
                  onClick={() => setIsEditing(true)}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg bg-cyan/10 hover:bg-cyan/20 text-cyan text-sm font-medium transition-all"
                >
                  <EditIcon />
                  Edit Profile
                </button>
              ) : (
                <div className="flex gap-2">
                  <button
                    onClick={() => {
                      setIsEditing(false)
                      setFormData({
                        fullName: user.fullName,
                        email: user.email,
                        company: user.company,
                        role: user.role,
                      })
                    }}
                    className="px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-white text-sm font-medium transition-all"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleSave}
                    className="px-4 py-2 rounded-lg bg-cyan hover:bg-cyan/90 text-dark text-sm font-medium transition-all"
                  >
                    Save Changes
                  </button>
                </div>
              )}
            </div>

            <div className="space-y-4">
              {/* Full Name */}
              <div>
                <label className="block text-sm font-medium text-white/60 mb-2">Full Name</label>
                {isEditing ? (
                  <input
                    type="text"
                    value={formData.fullName}
                    onChange={(e) => setFormData({ ...formData, fullName: e.target.value })}
                    className="w-full px-4 py-3 bg-dark border border-white/10 rounded-lg text-white focus:outline-none focus:border-cyan/50 focus:ring-2 focus:ring-cyan/20 transition-all"
                  />
                ) : (
                  <p className="text-white text-base">{user.fullName}</p>
                )}
              </div>

              {/* Email */}
              <div>
                <label className="block text-sm font-medium text-white/60 mb-2">Email Address</label>
                {isEditing ? (
                  <input
                    type="email"
                    value={formData.email}
                    onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                    className="w-full px-4 py-3 bg-dark border border-white/10 rounded-lg text-white focus:outline-none focus:border-cyan/50 focus:ring-2 focus:ring-cyan/20 transition-all"
                  />
                ) : (
                  <p className="text-white text-base">{user.email}</p>
                )}
              </div>

              {/* Company */}
              <div>
                <label className="block text-sm font-medium text-white/60 mb-2">Company</label>
                {isEditing ? (
                  <input
                    type="text"
                    value={formData.company}
                    onChange={(e) => setFormData({ ...formData, company: e.target.value })}
                    className="w-full px-4 py-3 bg-dark border border-white/10 rounded-lg text-white focus:outline-none focus:border-cyan/50 focus:ring-2 focus:ring-cyan/20 transition-all"
                  />
                ) : (
                  <p className="text-white text-base">{user.company}</p>
                )}
              </div>

              {/* Role */}
              <div>
                <label className="block text-sm font-medium text-white/60 mb-2">Role</label>
                {isEditing ? (
                  <select
                    value={formData.role}
                    onChange={(e) => setFormData({ ...formData, role: e.target.value })}
                    className="w-full px-4 py-3 bg-dark border border-white/10 rounded-lg text-white focus:outline-none focus:border-cyan/50 focus:ring-2 focus:ring-cyan/20 transition-all"
                  >
                    <option value="operator">Operator</option>
                    <option value="supervisor">Supervisor</option>
                    <option value="manager">Manager</option>
                    <option value="admin">Administrator</option>
                  </select>
                ) : (
                  <p className="text-white text-base capitalize">{user.role}</p>
                )}
              </div>
            </div>
          </motion.div>

          {/* Avatar Customization */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="rounded-2xl bg-card border border-white/5 p-6"
          >
            <h2 className="text-xl font-semibold text-white mb-4">Avatar Style</h2>
            <p className="text-white/60 text-sm mb-4">Choose your avatar style</p>
            <div className="grid grid-cols-3 gap-3">
              {avatarStyles.map((style) => (
                <button
                  key={style.value}
                  onClick={() => handleAvatarChange(style.value)}
                  className={`p-4 rounded-lg border-2 transition-all ${
                    selectedAvatarStyle === style.value
                      ? 'border-cyan bg-cyan/10'
                      : 'border-white/10 hover:border-white/20 bg-white/5'
                  }`}
                >
                  <div className="text-2xl mb-2">{style.preview}</div>
                  <p className="text-white text-xs font-medium">{style.name}</p>
                </button>
              ))}
            </div>
          </motion.div>

          {/* Theme Customization */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="rounded-2xl bg-card border border-white/5 p-6"
          >
            <h2 className="text-xl font-semibold text-white mb-4">Theme Color</h2>
            <p className="text-white/60 text-sm mb-4">Choose your accent color</p>
            <div className="grid grid-cols-6 gap-3">
              {themeColors.map((theme) => (
                <button
                  key={theme.value}
                  onClick={() => handleThemeChange(theme.value)}
                  className={`relative h-16 rounded-lg transition-all ${
                    selectedTheme === theme.value
                      ? 'ring-2 ring-white ring-offset-2 ring-offset-card scale-110'
                      : 'hover:scale-105'
                  }`}
                  style={{ backgroundColor: theme.color }}
                >
                  {selectedTheme === theme.value && (
                    <div className="absolute inset-0 flex items-center justify-center">
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3">
                        <path d="M5 13l4 4L19 7" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    </div>
                  )}
                </button>
              ))}
            </div>
          </motion.div>

          {/* Notifications */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="rounded-2xl bg-card border border-white/5 p-6"
          >
            <h2 className="text-xl font-semibold text-white mb-4">Notifications</h2>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-white text-sm font-medium">Email Notifications</p>
                  <p className="text-white/50 text-xs">Receive updates via email</p>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={notifications.email}
                    onChange={(e) => handleNotificationChange('email', e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-white/10 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-cyan"></div>
                </label>
              </div>
              
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-white text-sm font-medium">Push Notifications</p>
                  <p className="text-white/50 text-xs">Receive browser notifications</p>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={notifications.push}
                    onChange={(e) => handleNotificationChange('push', e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-white/10 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-cyan"></div>
                </label>
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <p className="text-white text-sm font-medium">Accident Alerts</p>
                  <p className="text-white/50 text-xs">Get notified of new accidents</p>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={notifications.accidents}
                    onChange={(e) => handleNotificationChange('accidents', e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-white/10 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-cyan"></div>
                </label>
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <p className="text-white text-sm font-medium">Camera Status</p>
                  <p className="text-white/50 text-xs">Alerts for camera issues</p>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={notifications.cameras}
                    onChange={(e) => handleNotificationChange('cameras', e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-white/10 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-cyan"></div>
                </label>
              </div>
            </div>
          </motion.div>

          {/* Security */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="rounded-2xl bg-card border border-white/5 p-6"
          >
            <h2 className="text-xl font-semibold text-white mb-4">Security</h2>
            <div className="space-y-3">
              <button className="w-full flex items-center justify-between p-4 rounded-lg bg-white/5 hover:bg-white/10 transition-all">
                <div className="text-left">
                  <p className="text-white text-sm font-medium">Change Password</p>
                  <p className="text-white/50 text-xs">Update your password regularly</p>
                </div>
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M6 4L10 8L6 12" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </button>
              <button className="w-full flex items-center justify-between p-4 rounded-lg bg-white/5 hover:bg-white/10 transition-all">
                <div className="text-left">
                  <p className="text-white text-sm font-medium">Two-Factor Authentication</p>
                  <p className="text-white/50 text-xs">Add an extra layer of security</p>
                </div>
                <div className="px-2 py-1 rounded bg-amber-500/20 text-amber-400 text-xs font-medium">
                  Disabled
                </div>
              </button>
            </div>
          </motion.div>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Avatar Card */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="rounded-2xl bg-card border border-white/5 p-6"
          >
            <div className="flex flex-col items-center text-center">
              <img 
                src={user.avatar} 
                alt={user.fullName}
                className="w-24 h-24 rounded-full mb-4 shadow-glow"
              />
              <h3 className="text-white text-lg font-semibold mb-1">{user.fullName}</h3>
              <p className="text-white/60 text-sm mb-1">{user.email}</p>
              <div className="px-3 py-1 rounded-full bg-cyan/10 text-cyan text-xs font-medium mt-2">
                {user.role.charAt(0).toUpperCase() + user.role.slice(1)}
              </div>
            </div>
          </motion.div>

          {/* Stats Card */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="rounded-2xl bg-card border border-white/5 p-6"
          >
            <h3 className="text-white font-semibold mb-4">Account Stats</h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-white/60 text-sm">Member since</span>
                <span className="text-white text-sm font-medium">
                  {new Date(user.createdAt).toLocaleDateString()}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-white/60 text-sm">Account age</span>
                <span className="text-white text-sm font-medium">
                  {accountAge} {accountAge === 1 ? 'day' : 'days'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-white/60 text-sm">User ID</span>
                <span className="text-white/40 text-xs font-mono">{user.id}</span>
              </div>
            </div>
          </motion.div>

          {/* Logout Button */}
          <motion.button
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            onClick={handleLogout}
            className="w-full flex items-center justify-center gap-2 p-4 rounded-xl bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 text-red-400 font-medium transition-all"
          >
            <LogoutIcon />
            Logout
          </motion.button>
        </div>
      </div>
    </div>
  )
}
