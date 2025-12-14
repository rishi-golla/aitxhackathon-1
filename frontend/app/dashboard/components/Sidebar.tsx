'use client'

import { usePathname } from 'next/navigation'
import Link from 'next/link'

const DashboardIcon = () => (
  <svg width="18" height="18" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
    <rect x="2" y="2" width="6" height="6" rx="1" />
    <rect x="12" y="2" width="6" height="6" rx="1" />
    <rect x="2" y="12" width="6" height="6" rx="1" />
    <rect x="12" y="12" width="6" height="6" rx="1" />
  </svg>
)

const LiveIcon = () => (
  <svg width="18" height="18" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
    <circle cx="10" cy="10" r="3" />
    <circle cx="10" cy="10" r="7" strokeDasharray="3 3" />
  </svg>
)

const AccidentsIcon = () => (
  <svg width="18" height="18" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M10 3L16 14H4L10 3Z" strokeLinejoin="round" />
    <path d="M10 8V11" strokeLinecap="round" />
    <circle cx="10" cy="13" r="0.5" fill="currentColor" />
  </svg>
)

const DemoIcon = () => (
  <svg width="18" height="18" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M8 2L2 5V11C2 14.5 4.5 17.5 8 18C11.5 17.5 14 14.5 14 11V5L8 2Z" strokeLinejoin="round" />
    <path d="M6 9L8 11L12 7" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
)

const VideoIcon = () => (
  <svg width="18" height="18" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
    <rect x="2" y="5" width="12" height="10" rx="1" />
    <path d="M14 8L18 6V14L14 12" strokeLinejoin="round" />
  </svg>
)

const CameraIcon = () => (
  <svg width="18" height="18" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
    <rect x="2" y="6" width="16" height="10" rx="1" />
    <circle cx="10" cy="11" r="3" />
    <path d="M6 6V5C6 4.44772 6.44772 4 7 4H13C13.5523 4 14 4.44772 14 5V6" />
  </svg>
)

const AIIcon = () => (
  <svg width="18" height="18" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M10 2V4M10 16V18M2 10H4M16 10H18M4.22 4.22L5.64 5.64M14.36 14.36L15.78 15.78M4.22 15.78L5.64 14.36M14.36 5.64L15.78 4.22" strokeLinecap="round" />
    <circle cx="10" cy="10" r="3" />
  </svg>
)

const SupportIcon = () => (
  <svg width="18" height="18" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
    <circle cx="10" cy="10" r="7" />
    <path d="M7 8.5C7 7.11929 8.11929 6 9.5 6C10.8807 6 12 7.11929 12 8.5C12 9.5 11 10 9.5 10.5V11.5" strokeLinecap="round" />
    <circle cx="9.5" cy="14" r="0.5" fill="currentColor" />
  </svg>
)

const SettingsIcon = () => (
  <svg width="18" height="18" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
    <circle cx="10" cy="10" r="2.5" />
    <path d="M10 2V4M10 16V18M2 10H4M16 10H18M4.22 4.22L5.64 5.64M14.36 14.36L15.78 15.78M4.22 15.78L5.64 14.36M14.36 5.64L15.78 4.22" strokeLinecap="round" />
  </svg>
)

const ProfileIcon = () => (
  <svg width="18" height="18" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
    <circle cx="10" cy="7" r="3" />
    <path d="M4 17C4 14.7909 5.79086 13 8 13H12C14.2091 13 16 14.7909 16 17" strokeLinecap="round" />
  </svg>
)

interface NavItemProps {
  label: string
  icon: React.ReactNode
  active?: boolean
  href: string
}

const NavItem = ({ label, icon, active, href }: NavItemProps) => (
  <Link
    href={href}
    className={`relative flex items-center gap-2.5 px-3 py-2 w-full text-left text-sm font-medium rounded-lg transition-all duration-200 group ${
      active 
        ? 'text-cyan bg-cyan/10 shadow-glow' 
        : 'text-white/60 hover:text-white hover:bg-white/5'
    }`}
  >
    {active && (
      <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-6 bg-cyan rounded-r-full shadow-glow" />
    )}
    <span className={`transition-all duration-200 ${active ? 'text-cyan' : 'text-white/60 group-hover:text-cyan'}`}>
      {icon}
    </span>
    <span>{label}</span>
  </Link>
)

export default function Sidebar() {
  const pathname = usePathname()
  
  return (
    <aside className="w-60 min-w-60 bg-dark-secondary h-screen flex flex-col border-r border-white/5">
      {/* Logo */}
      <div className="flex items-center gap-2.5 px-4 py-4">
        <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-cyan to-accent flex items-center justify-center shadow-glow">
          <svg width="18" height="18" viewBox="0 0 20 20" fill="none">
            <path d="M10 3L17 6.5V13.5L10 17L3 13.5V6.5L10 3Z" stroke="white" strokeWidth="1.5" strokeLinejoin="round"/>
            <path d="M10 7L13 9L10 11L7 9L10 7Z" fill="white"/>
          </svg>
        </div>
        <span className="text-white font-semibold text-sm">NDA Company</span>
      </div>
      
      {/* Main Navigation */}
      <nav className="flex-1 mt-1 px-2">
        <div className="space-y-0.5">
          <NavItem
            label="Dashboard"
            icon={<DashboardIcon />}
            active={pathname === '/dashboard'}
            href="/dashboard"
          />
          <NavItem
            label="Footage Review"
            icon={<VideoIcon />}
            active={pathname === '/dashboard/live'}
            href="/dashboard/live"
          />
          <NavItem
            label="Accidents"
            icon={<AccidentsIcon />}
            active={pathname === '/dashboard/accidents'}
            href="/dashboard/accidents"
          />
          <NavItem
            label="How it Works"
            icon={<SupportIcon />}
            active={pathname === '/dashboard/demo'}
            href="/dashboard/demo"
          />
        </div>
      </nav>
      
      {/* Divider */}
      <div className="border-t border-white/10 mx-3" />
      
      {/* Bottom Navigation */}
      <div className="py-3 px-2">
        <div className="space-y-0.5">
          <NavItem
            label="Profile"
            icon={<ProfileIcon />}
            active={pathname === '/dashboard/profile'}
            href="/dashboard/profile"
          />
        </div>
      </div>
    </aside>
  )
}
