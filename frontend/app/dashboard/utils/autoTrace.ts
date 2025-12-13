// Automatic wall detection from floor plan images

import { WallSegment } from '../types/floorplan'

export async function autoTraceWalls(imageUrl: string, floor: number): Promise<WallSegment[]> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    
    img.onload = () => {
      try {
        const canvas = document.createElement('canvas')
        const ctx = canvas.getContext('2d', { willReadFrequently: true })
        if (!ctx) {
          reject(new Error('Could not get canvas context'))
          return
        }

        // High resolution for better detection
        const targetSize = 1200
        const scale = Math.min(targetSize / img.width, targetSize / img.height)
        canvas.width = img.width * scale
        canvas.height = img.height * scale

        ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
        
        console.log('🔍 Auto-tracing walls from image...')
        const walls = detectWalls(imageData, canvas.width, canvas.height, floor)
        console.log(`✓ Detected ${walls.length} wall segments`)
        
        resolve(walls)
      } catch (error) {
        reject(error)
      }
    }
    
    img.onerror = () => reject(new Error('Failed to load image'))
    img.src = imageUrl
  })
}

function detectWalls(imageData: ImageData, width: number, height: number, floor: number): WallSegment[] {
  const data = imageData.data
  
  // Step 1: Convert to grayscale and enhance contrast
  const gray = new Uint8ClampedArray(width * height)
  let minGray = 255, maxGray = 0
  
  for (let i = 0; i < width * height; i++) {
    const idx = i * 4
    const g = Math.floor(0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2])
    gray[i] = g
    minGray = Math.min(minGray, g)
    maxGray = Math.max(maxGray, g)
  }
  
  // Normalize contrast
  const range = maxGray - minGray
  if (range > 0) {
    for (let i = 0; i < gray.length; i++) {
      gray[i] = Math.floor(((gray[i] - minGray) / range) * 255)
    }
  }
  
  // Step 2: Detect edges
  const edges = detectEdges(gray, width, height)
  
  // Step 3: Find lines using Hough transform
  const lines = findLines(edges, width, height)
  
  // Step 4: Convert to wall segments
  return linesToWalls(lines, width, height, floor)
}

function detectEdges(gray: Uint8ClampedArray, width: number, height: number): Uint8ClampedArray {
  const edges = new Uint8ClampedArray(width * height)
  const threshold = 40
  
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      // Sobel operator
      const gx = 
        -gray[(y - 1) * width + (x - 1)] + gray[(y - 1) * width + (x + 1)] +
        -2 * gray[y * width + (x - 1)] + 2 * gray[y * width + (x + 1)] +
        -gray[(y + 1) * width + (x - 1)] + gray[(y + 1) * width + (x + 1)]
      
      const gy =
        -gray[(y - 1) * width + (x - 1)] - 2 * gray[(y - 1) * width + x] - gray[(y - 1) * width + (x + 1)] +
        gray[(y + 1) * width + (x - 1)] + 2 * gray[(y + 1) * width + x] + gray[(y + 1) * width + (x + 1)]
      
      const magnitude = Math.sqrt(gx * gx + gy * gy)
      edges[y * width + x] = magnitude > threshold ? 255 : 0
    }
  }
  
  return edges
}

interface Line {
  x1: number
  y1: number
  x2: number
  y2: number
  strength: number
}

function findLines(edges: Uint8ClampedArray, width: number, height: number): Line[] {
  const lines: Line[] = []
  const minLineLength = Math.min(width, height) * 0.04
  const maxGap = 20
  
  // Horizontal lines
  for (let y = 0; y < height; y += 1) {
    let lineStart = -1
    let lastEdge = -1
    let edgeCount = 0
    
    for (let x = 0; x < width; x++) {
      let hasEdge = edges[y * width + x] === 255
      
      // Check nearby pixels for thick lines
      if (!hasEdge) {
        for (let dy = -2; dy <= 2; dy++) {
          if (y + dy >= 0 && y + dy < height && edges[(y + dy) * width + x] === 255) {
            hasEdge = true
            break
          }
        }
      }
      
      if (hasEdge) {
        if (lineStart === -1) lineStart = x
        lastEdge = x
        edgeCount++
      } else if (lineStart !== -1 && x - lastEdge > maxGap) {
        const length = lastEdge - lineStart
        if (length > minLineLength && edgeCount > length * 0.25) {
          lines.push({ x1: lineStart, y1: y, x2: lastEdge, y2: y, strength: edgeCount })
        }
        lineStart = -1
        edgeCount = 0
      }
    }
    
    if (lineStart !== -1) {
      const length = lastEdge - lineStart
      if (length > minLineLength && edgeCount > length * 0.25) {
        lines.push({ x1: lineStart, y1: y, x2: lastEdge, y2: y, strength: edgeCount })
      }
    }
  }
  
  // Vertical lines
  for (let x = 0; x < width; x += 1) {
    let lineStart = -1
    let lastEdge = -1
    let edgeCount = 0
    
    for (let y = 0; y < height; y++) {
      let hasEdge = edges[y * width + x] === 255
      
      // Check nearby pixels for thick lines
      if (!hasEdge) {
        for (let dx = -2; dx <= 2; dx++) {
          if (x + dx >= 0 && x + dx < width && edges[y * width + (x + dx)] === 255) {
            hasEdge = true
            break
          }
        }
      }
      
      if (hasEdge) {
        if (lineStart === -1) lineStart = y
        lastEdge = y
        edgeCount++
      } else if (lineStart !== -1 && y - lastEdge > maxGap) {
        const length = lastEdge - lineStart
        if (length > minLineLength && edgeCount > length * 0.25) {
          lines.push({ x1: x, y1: lineStart, x2: x, y2: lastEdge, strength: edgeCount })
        }
        lineStart = -1
        edgeCount = 0
      }
    }
    
    if (lineStart !== -1) {
      const length = lastEdge - lineStart
      if (length > minLineLength && edgeCount > length * 0.25) {
        lines.push({ x1: x, y1: lineStart, x2: x, y2: lastEdge, strength: edgeCount })
      }
    }
  }
  
  return mergeLines(lines)
}

function mergeLines(lines: Line[]): Line[] {
  const merged: Line[] = []
  const used = new Set<number>()
  const mergeThreshold = 10
  
  for (let i = 0; i < lines.length; i++) {
    if (used.has(i)) continue
    
    const line1 = lines[i]
    const isHorizontal = Math.abs(line1.y2 - line1.y1) < Math.abs(line1.x2 - line1.x1)
    
    let bestLine = line1
    used.add(i)
    
    for (let j = i + 1; j < lines.length; j++) {
      if (used.has(j)) continue
      
      const line2 = lines[j]
      const isHorizontal2 = Math.abs(line2.y2 - line2.y1) < Math.abs(line2.x2 - line2.x1)
      
      if (isHorizontal === isHorizontal2) {
        if (isHorizontal) {
          if (Math.abs(line1.y1 - line2.y1) < mergeThreshold) {
            const overlap = Math.min(bestLine.x2, line2.x2) - Math.max(bestLine.x1, line2.x1)
            if (overlap > -40) {
              bestLine = {
                x1: Math.min(bestLine.x1, line2.x1),
                y1: (bestLine.y1 + line2.y1) / 2,
                x2: Math.max(bestLine.x2, line2.x2),
                y2: (bestLine.y2 + line2.y2) / 2,
                strength: bestLine.strength + line2.strength,
              }
              used.add(j)
            }
          }
        } else {
          if (Math.abs(line1.x1 - line2.x1) < mergeThreshold) {
            const overlap = Math.min(bestLine.y2, line2.y2) - Math.max(bestLine.y1, line2.y1)
            if (overlap > -40) {
              bestLine = {
                x1: (bestLine.x1 + line2.x1) / 2,
                y1: Math.min(bestLine.y1, line2.y1),
                x2: (bestLine.x2 + line2.x2) / 2,
                y2: Math.max(bestLine.y2, line2.y2),
                strength: bestLine.strength + line2.strength,
              }
              used.add(j)
            }
          }
        }
      }
    }
    
    merged.push(bestLine)
  }
  
  // Sort by strength and keep top 500 (increased from 150)
  return merged.sort((a, b) => b.strength - a.strength).slice(0, 500)
}

function linesToWalls(lines: Line[], width: number, height: number, floor: number): WallSegment[] {
  const walls: WallSegment[] = []
  const worldScale = 14 / Math.max(width, height)
  const centerX = width / 2
  const centerY = height / 2
  
  for (const line of lines) {
    const worldX1 = (line.x1 - centerX) * worldScale
    const worldZ1 = (line.y1 - centerY) * worldScale
    const worldX2 = (line.x2 - centerX) * worldScale
    const worldZ2 = (line.y2 - centerY) * worldScale
    
    walls.push({
      id: `wall-auto-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      start: [worldX1, worldZ1],
      end: [worldX2, worldZ2],
      floor,
    })
  }
  
  return walls
}

