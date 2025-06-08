//
//  Contours.swift
//  roboflow-swift
//
//  Created by Maxwell Stone on 6/5/25.
//

import simd

// MARK: - Integer 2-d point
struct Point: Equatable { var x: Int; var y: Int }

// ============================================================
// 1. SIMD edge-map builder
// ============================================================
private func makeEdgeMap(w: Int, h: Int, mask: [UInt8]) -> [Bool] {
    var width = w
    var height = h
    if mask.count % width == 0 {
        height = mask.count / width
    } else if mask.count % (width + 1) == 0 {
        width += 1
        height = mask.count / (width)
    } else if mask.count % (width + 2) == 0 {
        width += 2
        height = mask.count / (width)
    }
    precondition(mask.count == width * height)
    var edge = Array(repeating: false, count: mask.count)

    // Helper to convert 2-d coords → flat index
    @inline(__always) func idx(_ x: Int, _ y: Int) -> Int { y * width + x }

    // Process one row at a time using 16-byte vectors
    let step = 16                   // SIMD16<UInt8>
    let simdZero = SIMD16<UInt8>(repeating: 0)

    for y in 0..<height {
        let rowStart = idx(0, y)
        let nextRow  = y + 1 < height ? idx(0, y + 1) : rowStart // clamp last

        // -- horizontal pass -------------------------------------------------
        var x = 0
        while x + step <= width {
            // load current, left, right, and down pixels
            let cur  = SIMD16<UInt8>(mask[rowStart + x ..< rowStart + x + step])
            let left = SIMD16<UInt8>(mask[rowStart + max(x-1,0) ..< rowStart + max(x-1,0) + step])
            let right = SIMD16<UInt8>(mask[rowStart + min(x+1,width-1) ..< rowStart + min(x+1,width-1) + step])
            let down = SIMD16<UInt8>(mask[nextRow  + x ..< nextRow  + x + step])

            // (cur != 0) & ((left == 0) | (right == 0) | (down == 0) | (up == 0))
            let isFG  = cur  .> simdZero               // SIMDMask<…>
            let leftBG  = left  .== simdZero
            let rightBG = right .== simdZero
            let downBG  = down  .== simdZero
            // let upBG    = up    .== simdZero   // if you include the row above

            // -------- use .|| (logical OR on masks)
            let neighBG = leftBG .| rightBG .| downBG           // .|| chains fine
            //                .|| upBG                            // add this if you computed `upBG`


            let isEdge  = isFG .& neighBG
            // Pack results into Bool array
            for i in 0..<step {
                edge[rowStart + x + i] = isEdge[i] != false
            }
            x += step
        }
        // remainder (<16)
        while x < width {
            let cur = mask[rowStart + x] != 0
            if cur {
                let bgLeft  = x == 0          || mask[rowStart + (x-1)] == 0
                let bgRight = x == width-1    || mask[rowStart + (x+1)] == 0
                let bgUp    = y == 0          || mask[rowStart - width + x] == 0
                let bgDown  = y == height-1   || mask[rowStart + width + x] == 0
                edge[rowStart + x] = bgLeft || bgRight || bgUp || bgDown
            }
            x += 1
        }
    }
    return edge
}

// ============================================================
// 2. Border-following (Moore neighbor) to polygons
// ============================================================
private func traceContours(width: Int,
                           height: Int,
                           edge: inout [Bool]) -> [[Point]] {

    @inline(__always) func idx(_ x: Int, _ y: Int) -> Int { y * width + x }

    let dirs = [ (1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1) ]
    var polys: [[Point]] = []

    for y in 0..<height {
        for x in 0..<width where edge[idx(x,y)] {
            var contour: [Point] = []
            var start = Point(x:x, y:y)
            var cur   = start
            var prevDir = 6             // start searching from NW

            repeat {
                contour.append(cur)
                edge[idx(cur.x, cur.y)] = false   // mark visited

                var found = false
                for s in 0..<8 {
                    let dir = (prevDir + 1 + s) & 7
                    let (dx, dy) = dirs[dir]
                    let nx = cur.x + dx, ny = cur.y + dy
                    if nx>=0, nx<width, ny>=0, ny<height, edge[idx(nx,ny)] {
                        cur = Point(x: nx, y: ny)
                        prevDir = (dir + 4) & 7
                        found = true
                        break
                    }
                }
                if !found { break }      // single-pixel edge
            } while cur != start
            if contour.first != contour.last { contour.append(contour.first!) }
            polys.append(contour)
        }
    }
    return polys
}

// ============================================================
// Public API
// ============================================================
struct SIMDContour {

    /// Returns one polygon per connected boundary in the mask.
    static func polygons(width: Int, height: Int, mask: [UInt8]) -> [[Point]] {
        var edge = makeEdgeMap(w: width, h: height, mask: mask)
        return traceContours(width: width, height: height, edge: &edge)
    }
}
