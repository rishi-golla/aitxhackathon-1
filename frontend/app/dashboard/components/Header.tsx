'use client'

import { useState } from 'react'

const ChevronDownIcon = () => (
  <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M3.5 5.25L7 8.75L10.5 5.25" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
)

const SearchIcon = () => (
  <svg width="18" height="18" viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5">
    <circle cx="8" cy="8" r="4" />
    <path d="M11 11L14 14" strokeLinecap="round" />
  </svg>
)

export default function Header() {
  const [searchFocused, setSearchFocused] = useState(false)
  
  return (
    <header className="flex justify-between items-center mb-2 animate-fade-in">
      <h1 className="text-3xl font-bold text-white">Dashboard</h1>
      
      <div className="flex items-center gap-3">
        {/* This Week Dropdown */}
        <button className="flex items-center gap-2 px-3 py-2 rounded-xl glass text-sm text-white/80 hover:text-white hover:border-cyan/30 border border-white/10 transition-all duration-200">
          <span>This week</span>
          <ChevronDownIcon />
        </button>
        
        {/* Search Bar */}
        <div className="relative">
          <input
            type="text"
            placeholder="Search"
            onFocus={() => setSearchFocused(true)}
            onBlur={() => setSearchFocused(false)}
            className={`w-44 bg-card text-white/80 text-sm px-3 py-2 rounded-xl pl-10 border transition-all duration-200 ${
              searchFocused 
                ? 'border-cyan shadow-glow' 
                : 'border-white/10 hover:border-white/20'
            } placeholder:text-white/40 focus:outline-none`}
          />
          <div className={`absolute left-3 top-1/2 -translate-y-1/2 transition-colors duration-200 ${
            searchFocused ? 'text-cyan' : 'text-white/50'
          }`}>
            <SearchIcon />
          </div>
        </div>
      </div>
    </header>
  )
}
