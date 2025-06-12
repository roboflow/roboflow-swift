//
//  MaskUtils.swift
//  roboflow-swift
//
//  Created by Maxwell Stone on 6/8/25.
//
import CoreML
import Foundation
import Accelerate
import Vision

// MARK: – Mask utilities  (ports of preprocess_segmentation_masks & friends) --
@available(iOS 18.0, macOS 15.0, *)
let gpuOnly      = MLComputePolicy(.cpuAndGPU)              // force GPU
@available(iOS 18.0, macOS 15.0, *)
let anePreferred = MLComputePolicy(.cpuAndNeuralEngine)   // CPU-fallback, prefer ANE
@available(iOS 18.0, macOS 15.0, *)
let anyFast      = MLComputePolicy(.all)                  // let Core ML choose

@available(iOS 18.0, macOS 15.0, *)
struct MaskUtils {
    
    /// Uses tensor math to project coefficients and protos into masks.
    static func preprocessSegmentationMasks(
        proto: MLMultiArray,
        protoShape: (c: Int, h: Int, w: Int),
        coeffs: [[Float]],                      // flat [nDet*c]
        procH: Int, procW: Int // processing height and width
    ) -> MLTensor? {
        let (c, mh, mw) = protoShape
        let shapedProto = MLShapedArray<Float>(proto)
        var masks = MLTensor(shapedProto)
        masks = masks.reshaped(to: [c, mh * mw])
        
        let rows = coeffs.count
        if (rows == 0) { return nil }
        let cols = coeffs.first!.count     // assume rectangular
        
        let flat = coeffs.flatMap { $0 }   // [Float] length = rows*cols
        
        let mlArray = try? MLMultiArray(
            shape: [rows as NSNumber, cols as NSNumber],
            dataType: .float32)
        
        mlArray?.dataPointer.withMemoryRebound(to: Float.self,
                                               capacity: flat.count) { ptr in
            flat.withUnsafeBufferPointer { src in
                ptr.update(from: src.baseAddress!, count: flat.count)
            }
        }
        
        let shapedCoeffs = MLShapedArray<Float>(mlArray!)
        
        masks = withMLTensorComputePolicy(anePreferred) {
            
            let coeffs_t = MLTensor(shapedCoeffs)
            var proc_masks = coeffs_t.matmul(masks)
            proc_masks = ((proc_masks * -1)
                .exp() + 1)
            .reciprocal() // sigmoid
            
            proc_masks = proc_masks.reshaped(to: [rows, mh, mw])
            
            proc_masks = proc_masks.resized(to: (procH, procW), method: .bilinear(alignCorners: false))
            return proc_masks
        }
        
        
        
        return masks
    }
    
    /// Full **accurate** path → returns binary mask resized to image
    static func processMaskAccurate(proto: MLMultiArray, protoShape: (Int,Int,Int),
                                    coeffs: [[Float]],
                                    dets: [CGRect],procH: Int, procW: Int) -> [[UInt8]] {
        // project protos and coefficients to masks
        guard let masks = preprocessSegmentationMasks(proto: proto,
                                                      protoShape: protoShape,
                                                      coeffs: coeffs,
                                                      procH: procH, procW: procW) else {
            return []
        }
        
        // crop each mask and convert to polygon
        let (h, w) = (masks.shape[1], masks.shape[2])
        let nDet = dets.count
        
        // Pre-allocate the result container so every worker writes
        // to a unique slot — no locking or .append() races.
        var result       = [] as [[UInt8]]
        
        for i in 0..<nDet {
            // 1. Grab detection-specific data --------------------------------------
            let det = dets[i]
            let x1  = Float(det.minX)
            let y1  = Float(det.minY)
            let x2  = Float(det.maxX)
            let y2  = Float(det.maxY)
            
            // 2. Mask maths ---------------------------------------------------------
            let thr:   Float = 0.5            // decision boundary
            // Integer bounds in tensor space
            let colStart = max(0, Int(floor(x1)))
            let colEnd   = min(w, Int(ceil(x2)))
            let rowStart = max(0, Int(floor(y1)))
            let rowEnd   = min(h, Int(ceil(y2)))
            
            let roiW = colEnd - colStart          // = x2 − x1
            let roiH = rowEnd - rowStart          // = y2 − y1
            
            
            let cropped = withMLTensorComputePolicy(anePreferred) {
                let mask = masks[i]                    // MLTensor  [h,w]
                
                // Build index tensors once per detection -------------------------------
                let rowIdxTensor: MLTensor = {
                    let rows = (rowStart ..< rowEnd).map { Int32($0) }
                    return MLTensor(MLShapedArray<Int32>(scalars: rows, shape: [roiH]))
                }()
                
                let colIdxTensor: MLTensor = {
                    let cols = (colStart ..< colEnd).map { Int32($0) }
                    return MLTensor(MLShapedArray<Int32>(scalars: cols, shape: [roiW]))
                }()
                
                // crop the mask so that only pixels within the detection box boundsa are kept
                let rows = mask.gathering(atIndices: rowIdxTensor, alongAxis: 0) // slice height
                let roi  = rows.gathering(atIndices: colIdxTensor, alongAxis: 1) // slice width
                return (roi .> thr) * Float(255)                             // threshold and rescale
            }
            
            // 3. Convert to scalars (still blocking; dominates runtime) ------------
            let shaped: MLShapedArray<Float> = try! shapedArraySync(cropped,
                                                                    as: Float.self)
            
            let elemCount = roiW * roiH                 // final buffer size
            
            // ──────────────────────────────────────────────────────────────
            // 7.  SIMD cast  Float → UInt8  **only inside the ROI**
            // ──────────────────────────────────────────────────────────────
            var uint8Flat = [UInt8](repeating: 0, count: elemCount)
            
            shaped.scalars.withUnsafeBufferPointer { src in
                uint8Flat.withUnsafeMutableBufferPointer { dst in
                    vDSP_vfixu8(src.baseAddress!, 1,
                                dst.baseAddress!, 1,
                                vDSP_Length(elemCount))
                }
            }
            
            // 5. Store into the pre-allocated slot ----------------------------------
            result.append(uint8Flat as [UInt8])
        }
        
        return result
    }
    
    // MARK: – Greedy NMS (non_max_suppression_fast) ------------------------------
    static func nonMaxSuppressionFast(_ dets: [[Float]], iouThresh: Float) -> [[Float]] {
        guard !dets.isEmpty else { return [] }
        var boxes = dets
        // sort by confidence
        boxes.sort { $0[4] > $1[4] }
        var keep = [[Float]]()
        
        while !boxes.isEmpty {
            let a = boxes.removeFirst()
            keep.append(a)
            boxes = boxes.filter { b in
                let iou = iouRectIoU(a,b)
                return iou <= iouThresh
            }
        }
        return keep
    }
    
    /// calculate the IoU of two boxes
    static func iouRectIoU(_ a:[Float], _ b:[Float]) -> Float {
        let xA = max(a[0], b[0]), yA = max(a[1], b[1])
        let xB = min(a[2], b[2]), yB = min(a[3], b[3])
        let interW = max(0, xB - xA), interH = max(0, yB - yA)
        let inter = interW * interH
        let areaA = (a[2]-a[0]) * (a[3]-a[1])
        let areaB = (b[2]-b[0]) * (b[3]-b[1])
        return inter / (areaA + areaB - inter)
    }
    
    /// parse a binary mask (0 or 255) to a set of border points that create a polygon outlining the mask's contour
    static func maskToPolygons(
        m: [UInt8],
        width: Int,
        h: Int) throws -> [[CGPoint]] {
        var mask = m
        var height = h
        if width * height != mask.count {
            return []
        }
        
        // 1️⃣ Binary data ➜ one‑channel CGImage ------------------------------
        if height < 64 && width < 64 {
            // need to pad
            let pad: [UInt8] = Array(repeating: 0, count: width)
            for _ in 0..<64-height {
                mask.append(contentsOf: pad)
            }
            height = 64
        }
        let cfData   = CFDataCreate(nil, mask, mask.count)!
        let provider = CGDataProvider(data: cfData)!
        let cgMask   = CGImage(width:  width,
                               height: height,
                               bitsPerComponent: 8,
                               bitsPerPixel: 8,
                               bytesPerRow: width,
                               space: CGColorSpaceCreateDeviceGray(),
                               bitmapInfo: [],                   // no alpha
                               provider: provider,
                               decode: nil,
                               shouldInterpolate: false,
                               intent: .defaultIntent)!
        
        
        // 2️⃣ Vision contour detection ---------------------------------------
        let request = VNDetectContoursRequest()
        request.detectsDarkOnLight   = false   // mask = white foreground
        request.contrastAdjustment   = 1.0     // no stretching needed
        request.maximumImageDimension = max(width, height)  // keep full res
        
        try VNImageRequestHandler(cgImage: cgMask, options: [:])
            .perform([request])
        
        guard let obs = request.results?.first as? VNContoursObservation else {
            return []
        }
        
        // 3️⃣ Copy contours + denormalise y‑axis (Vision y: 0=bottom, UIKit 0=top)
        var polygons = [[CGPoint]]()
        for i in 0..<obs.topLevelContourCount {
            let contour = obs.topLevelContours[i]
            let ring = contour.normalizedPoints.map { p in
                CGPoint(x: CGFloat(p.x) * CGFloat(width),
                        y: (1 - CGFloat(p.y)) * CGFloat(height))
            }
            polygons.append(ring)
        }
        return polygons
    }
}

/// materialize a MLTensor to a MLShapedArray for further processing
@available(iOS 18.0, macOS 15.0, *)
func shapedArraySync<T: MLTensorScalar>(
    _ tensor: MLTensor,
    as _: T.Type = T.self) throws -> MLShapedArray<T> {
    let sema = DispatchSemaphore(value: 0)
    var result: Result<MLShapedArray<T>, Error>!
    
    Task(priority: .userInitiated) {
        let sa = await tensor.shapedArray(of: T.self)
        result = .success(sa)
        sema.signal()
    }
    
    sema.wait()
    return try result.get()
}
