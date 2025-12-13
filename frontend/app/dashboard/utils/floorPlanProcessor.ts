// Advanced Floor Plan Image Processing Utilities

export interface DetectedRoom {
  position: [number, number, number]
  size: [number, number, number]
  color: string
  opacity: number
  type: 'production' | 'warehouse' | 'common' | 'secure'
}

export interface DetectedWall {
  position: [number, number, number]
  size: [number, number, number]
}

export interface ProcessedFloorPlan {
  rooms: DetectedRoom[]
  walls: DetectedWall[]
  scale: number
}

/**
 * Process an uploaded floor plan image with high-resolution contour detection
 */
export async function processFloorPlanImage(
  imageUrl: string
): Promise<ProcessedFloorPlan> {
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

        // Use higher resolution for better detail
        const maxSize = 1024
        const scale = Math.min(maxSize / img.width, maxSize / img.height)
        canvas.width = img.width * scale
        canvas.height = img.height * scale

        ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
        
        const result = analyzeFloorPlanAdvanced(imageData, canvas.width, canvas.height)
        
        resolve(result)
      } catch (error) {
        reject(error)
      }
    }
    
    img.onerror = () => reject(new Error('Failed to load image'))
    img.src = imageUrl
  })
}

/**
 * Advanced floor plan analysis with contour detection
 */
function analyzeFloorPlanAdvanced(
  imageData: ImageData,
  width: number,
  height: number
): ProcessedFloorPlan {
  const data = imageData.data
  
  // Step 1: Convert to grayscale and detect edges
  const edges = detectEdgesCanny(data, width, height)
  
  // Step 2: Find contours (closed shapes)
  const contours = findContours(edges, width, height)
  
  // Step 3: Classify contours as rooms or walls
  const { rooms, walls } = classifyContours(contours, data, width, height)
  
  return {
    rooms,
    walls,
    scale: 16 / Math.max(width, height),
  }
}

/**
 * Canny edge detection
 */
function detectEdgesCanny(
  data: Uint8ClampedArray,
  width: number,
  height: number
): Uint8ClampedArray {
  const edges = new Uint8ClampedArray(width * height)
  
  // Convert to grayscale
  const gray = new Uint8ClampedArray(width * height)
  for (let i = 0; i < width * height; i++) {
    const idx = i * 4
    gray[i] = Math.floor(0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2])
  }
  
  // Sobel edge detection
  const sobelX = [-1, 0, 1, -2, 0, 2, -1, 0, 1]
  const sobelY = [-1, -2, -1, 0, 0, 0, 1, 2, 1]
  
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let gx = 0, gy = 0
      
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const idx = (y + ky) * width + (x + kx)
          const kidx = (ky + 1) * 3 + (kx + 1)
          gx += gray[idx] * sobelX[kidx]
          gy += gray[idx] * sobelY[kidx]
        }
      }
      
      const magnitude = Math.sqrt(gx * gx + gy * gy)
      edges[y * width + x] = magnitude > 50 ? 255 : 0
    }
  }
  
  return edges
}

/**
 * Find contours using connected component analysis
 */
interface Contour {
  points: [number, number][]
  bounds: { minX: number; maxX: number; minY: number; maxY: number }
  area: number
}

function findContours(
  edges: Uint8ClampedArray,
  width: number,
  height: number
): Contour[] {
  const visited = new Uint8ClampedArray(width * height)
  const contours: Contour[] = []
  
  // Scan for regions
  for (let y = 0; y < height; y += 4) { // Sample every 4 pixels for performance
    for (let x = 0; x < width; x += 4) {
      const idx = y * width + x
      if (!visited[idx] && edges[idx] === 0) { // Look for non-edge regions (rooms)
        const contour = floodFill(edges, visited, x, y, width, height)
        if (contour.area > 100) { // Filter small noise
          contours.push(contour)
        }
      }
    }
  }
  
  return contours
}

/**
 * Flood fill to find connected regions
 */
function floodFill(
  edges: Uint8ClampedArray,
  visited: Uint8ClampedArray,
  startX: number,
  startY: number,
  width: number,
  height: number
): Contour {
  const points: [number, number][] = []
  const stack: [number, number][] = [[startX, startY]]
  
  let minX = startX, maxX = startX, minY = startY, maxY = startY
  
  while (stack.length > 0) {
    const [x, y] = stack.pop()!
    const idx = y * width + x
    
    if (x < 0 || x >= width || y < 0 || y >= height) continue
    if (visited[idx] || edges[idx] === 255) continue
    
    visited[idx] = 1
    points.push([x, y])
    
    minX = Math.min(minX, x)
    maxX = Math.max(maxX, x)
    minY = Math.min(minY, y)
    maxY = Math.max(maxY, y)
    
    // 4-connected neighbors
    if (points.length < 10000) { // Limit to prevent infinite loops
      stack.push([x + 4, y])
      stack.push([x - 4, y])
      stack.push([x, y + 4])
      stack.push([x, y - 4])
    }
  }
  
  return {
    points,
    bounds: { minX, maxX, minY, maxY },
    area: points.length,
  }
}

/**
 * Classify contours as rooms or walls based on their properties
 */
function classifyContours(
  contours: Contour[],
  data: Uint8ClampedArray,
  width: number,
  height: number
): { rooms: DetectedRoom[]; walls: DetectedWall[] } {
  const rooms: DetectedRoom[] = []
  const walls: DetectedWall[] = []
  
  // World scale: map image to 16x16 world units
  const worldScale = 16 / Math.max(width, height)
  const centerX = width / 2
  const centerY = height / 2
  
  for (const contour of contours) {
    const { bounds, area, points } = contour
    
    // Calculate dimensions
    const w = bounds.maxX - bounds.minX
    const h = bounds.maxY - bounds.minY
    const aspectRatio = w / h
    
    // Sample color from center of region
    const centerIdx = ((bounds.minY + bounds.maxY) / 2 * width + (bounds.minX + bounds.maxX) / 2) * 4
    const r = data[Math.floor(centerIdx)]
    const g = data[Math.floor(centerIdx + 1)]
    const b = data[Math.floor(centerIdx + 2)]
    
    // Convert to world coordinates
    const worldX = ((bounds.minX + bounds.maxX) / 2 - centerX) * worldScale
    const worldZ = ((bounds.minY + bounds.maxY) / 2 - centerY) * worldScale
    const worldW = w * worldScale
    const worldH = h * worldScale
    
    // Classify as wall if very thin
    if ((aspectRatio > 8 || aspectRatio < 0.125) && area < 5000) {
      walls.push({
        position: [worldX, 0.5, worldZ],
        size: aspectRatio > 1 ? [worldW, 1, 0.15] : [0.15, 1, worldH],
      })
    } else if (area > 200) {
      // Classify as room
      const { type, color, opacity } = getRoomPropertiesFromColor(r, g, b)
      
      rooms.push({
        position: [worldX, 0.3, worldZ],
        size: [worldW, 0.6, worldH],
        color,
        opacity,
        type,
      })
    }
  }
  
  return { rooms, walls }
}

/**
 * Determine room properties from RGB color
 */
function getRoomPropertiesFromColor(
  r: number,
  g: number,
  b: number
): { type: DetectedRoom['type']; color: string; opacity: number } {
  const brightness = (r + g + b) / 3
  
  // Very dark or very light = likely background/walls, use neutral
  if (brightness < 40 || brightness > 240) {
    return { type: 'common', color: '#6B7280', opacity: 0.25 }
  }
  
  // Blue dominant
  if (b > r + 30 && b > g + 30) {
    return { type: 'production', color: '#5B8FF9', opacity: 0.5 }
  }
  
  // Purple
  if (r > 100 && b > 100 && Math.abs(r - b) < 60 && g < r - 20) {
    return { type: 'warehouse', color: '#7C3AED', opacity: 0.45 }
  }
  
  // Cyan/Teal
  if (g > r + 20 && b > r + 20) {
    return { type: 'common', color: '#3ddbd9', opacity: 0.4 }
  }
  
  // Green
  if (g > r + 30 && g > b + 30) {
    return { type: 'common', color: '#10B981', opacity: 0.4 }
  }
  
  // Red/Orange
  if (r > g + 30 && r > b + 30) {
    return { type: 'secure', color: '#EF4444', opacity: 0.4 }
  }
  
  // Pink/Magenta
  if (r > 150 && b > 100 && g < r - 30) {
    return { type: 'secure', color: '#EC4899', opacity: 0.4 }
  }
  
  // Yellow/Beige
  if (r > 150 && g > 150 && b < 150) {
    return { type: 'common', color: '#F59E0B', opacity: 0.35 }
  }
  
  // Default: use detected color with slight transparency
  const hexColor = `#${[r, g, b].map(c => Math.floor(c).toString(16).padStart(2, '0')).join('')}`
  return { type: 'common', color: hexColor, opacity: 0.35 }
}
