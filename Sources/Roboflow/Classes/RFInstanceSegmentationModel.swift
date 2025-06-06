//
//  RFObjectDetectionModel.swift
//  Roboflow
//
//  Created by Nicholas Arner on 4/12/22.
//

import Foundation
import CoreML
import Vision
import UIKit
import Accelerate

enum MaskProcessingMode {
    case quality
    case balanced
    case performance
}

@available(iOS 18.0, *)
let gpuOnly      = MLComputePolicy(.cpuAndGPU)              // force GPU
@available(iOS 18.0, *)
let anePreferred = MLComputePolicy(.cpuAndNeuralEngine)   // CPU-fallback, prefer ANE
@available(iOS 18.0, *)
let anyFast      = MLComputePolicy(.all)                  // let Core ML choose

//Creates an instance of an ML model that's hosted on Roboflow
public class RFInstanceSegmentationModel: RFObjectDetectionModel {
    var classes = [String]()
    var maskProcessingMode: MaskProcessingMode = .balanced
    //Load the retrieved CoreML model into an already created RFObjectDetectionModel instance
    override func loadMLModel(modelPath: URL, colors: [String: String], classes: [String]) -> Error? {
        print("loading instance seg coreml model")
        self.colors = colors
        self.classes = classes
        do {
            let config = MLModelConfiguration()
            if #available(iOS 16.0, *) {
                config.computeUnits = .cpuAndNeuralEngine
                print("add cpu fallback")
            } else {
                // Fallback on earlier versions
            }
            print("created model config")
            mlModel = try yolov8_seg(contentsOf: modelPath, configuration: config).model
            print("loaded model class")
            visionModel = try VNCoreMLModel(for: mlModel)
            print("created VNCoreMLModel")
            visionModel.featureProvider = super.thresholdProvider
            let request = VNCoreMLRequest(model: visionModel)
            request.imageCropAndScaleOption = .scaleFill
            coreMLRequest = request
            print("Created coreML request")
        } catch {
            return error
        }
        return nil
    }
    
    //Run image through model and return Detections
    @available(*, renamed: "detect(image:)")
    public override func detect(image:UIImage, completion: @escaping (([RFObjectDetectionPrediction]?, Error?) -> Void)) {
        let imgHeight = CGFloat(image.size.height)
        let imgWidth = CGFloat(image.size.width)
        
        let outputSize = self.maskProcessingMode == .balanced ? CGSize(width: 640, height: 640) : CGSize(width: imgWidth, height: imgHeight)
        guard let coreMLRequest = self.coreMLRequest else {
            completion(nil, "Model initialization failed.")
            return
        }
        guard let ciImage = CIImage(image: image) else {
            completion(nil, "Image failed.")
            return
        }
        
        // resize image to input dimmensions
        let resizeFilter = CIFilter(name:"CILanczosScaleTransform")!

        // Desired output size
        let targetSize = CGSize(width: 640, height: 640)

        // Compute scale and corrective aspect ratio
        let scale = targetSize.height / (ciImage.extent.height)
        let aspectRatio = targetSize.width/((ciImage.extent.width) * scale)

        // Apply resizing
        resizeFilter.setValue(ciImage, forKey: kCIInputImageKey)
        resizeFilter.setValue(scale, forKey: kCIInputScaleKey)
        resizeFilter.setValue(aspectRatio, forKey: kCIInputAspectRatioKey)
        let outputImage = resizeFilter.outputImage
        
        let handler = VNImageRequestHandler(ciImage: outputImage!, options: [:])

        do {
            let t0 = Date()
            try handler.perform([coreMLRequest])
            guard let detectResults = coreMLRequest.results else { return }
            
            // print("Model Inference Time: \(Date().timeIntervalSince(t0)) seconds")
            let t1 = Date()
            
            let predictions = detectResults[1] as! VNCoreMLFeatureValueObservation
            let protos = detectResults[0] as! VNCoreMLFeatureValueObservation
            
            let pred = predictions.featureValue.multiArrayValue!
            let proto = protos.featureValue.multiArrayValue!
            
            
            let numMasks = 32
            let numCls = self.colors.count
            // TODO:
            // - run NMS on predictions
            
            
            // --- flatten MLMultiArray to Swift [Float] for speed
            let p = pred.dataPointer.bindMemory(to: Float.self,
                                                capacity: pred.count)
            let preds = UnsafeBufferPointer(start: p, count: pred.count)
            let protoShape = (c:Int(truncating: proto.shape[1]),
                              h:Int(truncating: proto.shape[2]),
                              w:Int(truncating: proto.shape[3]))
            
            // each row = 5 + numCls + numMasks
            let stride = 4 + numCls + numMasks
            let numDet = preds.count / stride
            
            var detRows = [[Float]]()
            var coeffs  = [[Float]]()
            
            // print("multi array processing took: \(Date().timeIntervalSince(t1)) seconds")
            let t2a = Date()
                    
            // MARK: -- constants and helpers
            let spatial   = numDet                         // number of rows in a “column block”

            // pre-allocate results to avoid repeated realloc
            detRows.reserveCapacity(pred.shape[2].intValue)
            coeffs .reserveCapacity(pred.shape[2].intValue)

            // flatten MLMultiArray to raw Float pointer
            let basePtr = pred.dataPointer.assumingMemoryBound(to: Float.self)

            // lock for the result buffers (used only when we keep a detection)
            let outLock = NSLock()

            // MARK: -- parallel pass over detections
            DispatchQueue.concurrentPerform(iterations: numDet) { i in
                // ---- read bbox (cx,cy,w,h)  ----
                @inline(__always) func col(_ k: Int) -> Float {
                    basePtr[k * spatial + i]       // fast pointer math, no multiply in inner loops
                }
                let cx = col(0), cy = col(1)
                let w  = col(2), h  = col(3)

                // ---- arg-max over class scores ----
                var bestScore: Float = 0
                var bestCls  : Int   = -1
                var k = 4                         // first class score column
                while k < 4 + numCls {
                    let s = col(k)
                    if s > bestScore { bestScore = s; bestCls = k-4 }
                    k &+= 1
                }
                guard bestScore >= Float(threshold) else { return }   // prunes most rows quickly

                // ---- collect mask coefficients ----
                var localCoeff = [Float](repeating: 0, count: numMasks)
                var cidx = 4 + numCls               // first coeff column
                for m in 0..<numMasks {
                    localCoeff[m] = col(cidx)
                    cidx &+= 1
                }

                // ---- xywh → xyxy  ----
                let halfW = w * 0.5, halfH = h * 0.5
                let bbox  = simd_float4(cx - halfW,    // x1
                                        cy - halfH,    // y1
                                        cx + halfW,    // x2
                                        cy + halfH)    // y2

                // ---- append to outputs (thread-safe) ----
                outLock.lock()
                detRows.append([bbox.x, bbox.y, bbox.z, bbox.w, bestScore, Float(bestCls)])
                coeffs .append(localCoeff)
                outLock.unlock()
            }

            
            // print("Model output format processing took: \(Date().timeIntervalSince(t2a)) seconds")
            let t2 = Date()
            
            var kept: [[Float]] = []
            if #available(iOS 18.0, *) {
                kept = MaskUtils.nonMaxSuppressionFast(detRows, iouThresh: Float(self.overlap))
            } else {
                // Fallback on earlier versions
                kept = [] as [[Float]]
            }
            var final: [RFObjectDetectionPrediction] = []
            
            var boxesKeep = [CGRect]()
            var coeffsKeep = [[Float]]()
            var keeps = [[Float]]()
            
            // gather masks only for kept indices
            for keep in kept {
                guard let idx = detRows.firstIndex(where: { $0.elementsEqual(keep) }) else { continue }
                
                let width = (keep[2] - keep[0])
                let height = (keep[3] - keep[1])
                let minX = CGFloat(keep[0])
                let minY = CGFloat(keep[1])
                let xs = imgWidth / 640
                let ys = imgHeight / 640
                let box = CGRect(x: CGFloat(minX*xs), y: CGFloat(minY*ys), width: CGFloat(width)*xs, height: CGFloat(height)*ys)
                
                boxesKeep.append(box)
                coeffsKeep.append(coeffs[idx])
                keeps.append(keep)
                
            }
            
            // print("NMS took: \(Date().timeIntervalSince(t2)) seconds")
            let t3 = Date()
            if #available(iOS 18.0, *) {
                let maskBins = MaskUtils.processMaskAccurate(proto: proto,
                                                             protoShape: protoShape,
                                                             coeffs: coeffsKeep,
                                                             dets: boxesKeep,
                                                             imgH: Int(image.size.height), imgW: Int(image.size.width),
                                                             procH: Int(outputSize.height), procW: Int(outputSize.width))
                print("time to process masks: \(Date().timeIntervalSince(t3)) seconds")
                let t4 = Date()
                
                
                for (i, maskBin) in maskBins.enumerated() {
                    let flatMask: [UInt8] = maskBin     // cols
                    let box = boxesKeep[i]
                    let timeToPolygon = Date()
                    
                    let x1  = Float(box.minX)
                    let y1  = Float(box.minY)
                    let x2  = Float(box.maxX)
                    let y2  = Float(box.maxY)

                    // Integer bounds in tensor space
                    let colStart = max(0, Int(floor(x1)))
                    let colEnd   = min(Int(imgWidth), Int(ceil(x2))) as Int
                    let rowStart = max(0, Int(floor(y1)))
                    let rowEnd   = min(Int(imgHeight), Int(ceil(y2)))

                    let roiW = colEnd - colStart          // = x2 − x1
                    let roiH = rowEnd - rowStart          // = y2 − y1
                    print("Width * Height: \(roiW * roiH); num pixels: \(flatMask.count)")
                    var polys = [[CGPoint]]()
                    do {
                        polys = try MaskUtils.maskToPolygons(m: flatMask, width: roiW, h: roiH)
                    } catch {
                        print(error)
                    }
//                    let polys = SIMDContour.polygons(width: Int(box.width), height: Int(box.height), mask: flatMask)
                    // print("time to polygon: \(Date().timeIntervalSince(timeToPolygon)) seconds")
                    
                    var polygon = (polys.count ?? 0 > 0 ? polys[0] : []) ?? []
                    
                    if polygon.count == 0 {
                        print("no polygons returned")
                        print("polys: \(polys ?? [[]])")
                    }
                    
                    polygon = polygon.map { CGPoint(x: CGFloat($0.x + box.minX), y: CGFloat($0.y + box.minY)) }
                    
                    let keep = keeps[i]
                    
                    let classname = classes[Int(keep[5])]
                    let detection = RFInstanceSegmentationPrediction(x: Float(box.midX), y: Float(box.midY), width: Float(box.width), height: Float(box.height), className: classname, confidence: keep[4], color: hexStringToUIColor(hex: colors[classname] ?? "#ff0000"), box: box, points:polygon, mask: nil)
                    final.append(detection)
                }
                print("Formulating Predictions Took: \(Date().timeIntervalSince(t4)) seconds")
            }
            
            print("complete Mask Processing Took: \(Date().timeIntervalSince(t3)) seconds")
            
            completion(final, nil)
        } catch let error {
            completion(nil, error)
        }
    }
    
    public override func detect(image: UIImage) async -> ([RFPrediction]?, Error?) {
        return await withCheckedContinuation { continuation in
            detect(image: image) { result, error in
                continuation.resume(returning: (result, error))
            }
        }
    }
    
    public override func detect(pixelBuffer: CVPixelBuffer, completion: @escaping (([RFObjectDetectionPrediction]?, Error?) -> Void)) {
        let image = UIImage(pixelBuffer: pixelBuffer)
        detect(image: image!) { detections, error in
            completion(detections, nil)
        }
    }
 
    public override func detect(pixelBuffer: CVPixelBuffer) async -> ([RFPrediction]?, Error?) {
        return await withCheckedContinuation { continuation in
            detect(pixelBuffer: pixelBuffer) { result, error in
                continuation.resume(returning: (result, error))
            }
        }
    }
    


    // ----------------------------------------------------------------------------

    // MARK: – Mask utilities  (ports of preprocess_segmentation_masks & friends) --
    @available(iOS 18.0, *)
    struct MaskUtils {
        
        /// Builds instance-segmentation masks without using Accelerate.
        static func preprocessSegmentationMasks(
            proto: MLMultiArray,
            protoShape: (c: Int, h: Int, w: Int),
            coeffs: [[Float]],                // flat [nDet*c]
            outH: Int, outW: Int
        ) -> MLTensor? {                    // one crop per detection
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
                
                proc_masks = proc_masks.resized(to: (outH, outW), method: .bilinear(alignCorners: false))
                return proc_masks
            }
            
            
            
            return masks
        }
        
        
        static func cropMask(_ mask: inout [Float],
                             det: CGRect,
                             maskW: Int,
                             maskH: Int) {
            // Zero-out pixels outside det (exact port of crop_mask)
            let x1 = Int(det.minX.rounded(.down))
            let y1 = Int(det.minY.rounded(.down))
            let x2 = Int(det.maxX.rounded(.up))
            let y2 = Int(det.maxY.rounded(.up))
            
            for y in 0..<maskH {
                for x in 0..<maskW {
                    if x < x1 || x >= x2 || y < y1 || y >= y2 {
                        mask[y*maskW + x] = 0
                    }
                }
            }
        }
        
        /// Full **accurate** path → returns binary mask resized to image
        static func processMaskAccurate(proto: MLMultiArray, protoShape: (Int,Int,Int),
                                                coeffs: [[Float]],
                                                dets: [CGRect],
                                                imgH: Int,
                                        imgW: Int, procH: Int, procW: Int) -> [[UInt8]] {
            let t0 = Date()
                    guard let masks = preprocessSegmentationMasks(proto: proto,
                                                               protoShape: protoShape,
                                                               coeffs: coeffs,
                                                                     outH: imgH, outW: imgW) else {
                        return []
                    }
            
                    
                    let (h, w) = (masks.shape[1], masks.shape[2])
                    let nDet = dets.count

                    let rowVector = (0..<w).map { Float($0) }
                    let colVector = (0..<h).map { Float($0) }

//                    let r = MLTensor(MLShapedArray<Float>(scalars: rowVector, shape: [w]))
//                                .reshaped(to: [1, 1, w])
//
//                    let c = MLTensor(MLShapedArray<Float>(scalars: colVector, shape: [h]))
//                                .reshaped(to: [1, h, 1])

            print("time to do math: \(Date().timeIntervalSince(t0))")
            let t1 = Date()
            // Pre-allocate the result container so every worker writes
            // to a unique slot — no locking or .append() races.
            var result       = []as [[UInt8]]
            let resultLock   = NSLock()          // only for stderr / print, not result.

            // ---------------------------------------------------------------------------
            //  Parallel pass — one iteration per detection
            // ---------------------------------------------------------------------------
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
                    let rotatedMask = masks[i]                    // MLTensor  [h,w]
//                    let keep  =  c .< y2 .& c .> y1 .& r .< x2 .& r .> x1
                    
                    // Build index tensors once per detection -------------------------------
                    let rowIdxTensor: MLTensor = {
                        let rows = (rowStart ..< rowEnd).map { Int32($0) }
                        return MLTensor(MLShapedArray<Int32>(scalars: rows, shape: [roiH]))
                    }()

                    let colIdxTensor: MLTensor = {
                        let cols = (colStart ..< colEnd).map { Int32($0) }
                        print(cols)
                        return MLTensor(MLShapedArray<Int32>(scalars: cols, shape: [roiW]))
                    }()
                    
                    let rows   = rotatedMask.gathering(atIndices: rowIdxTensor, alongAxis: 0)   // (1)
                        let roi    = rows.gathering(atIndices: colIdxTensor, alongAxis: 1)          // (2)
                        return (roi .> thr) * Float(255)                             // 0 / 255  (Float)
//                        .cast(to: UInt8.self)                  // change dtype so we fetch UInt8s
                }

                // 3. Convert to scalars (still blocking; dominates runtime) ------------
                let shapedSyncTime = Date()
                let shaped: MLShapedArray<Float> = try! shapedArraySync(cropped,
                                                                        as: Float.self)
                
//                let scalars = shaped.scalars                  // flat [Float]
                print("time to sync shape: \(Date().timeIntervalSince(shapedSyncTime))")
                let beforeUint8Time = Date()

                let elemCount = roiW * roiH                 // final buffer size
                print("elements: \(elemCount) \(shaped.scalars.count)")
                
                let sum = shaped.scalars.map { Int($0) }.reduce(0, +)
                print("sum tensor: \(sum)")

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

                
                 print("time to convert to uint8: \(Date().timeIntervalSince(beforeUint8Time))")

                // 5. Store into the pre-allocated slot ----------------------------------
                result.append(uint8Flat as [UInt8])
            }
            print("time to convert to usable format: \(Date().timeIntervalSince(t1))")
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
        
        static func iouRectIoU(_ a:[Float], _ b:[Float]) -> Float {
            let xA = max(a[0], b[0]), yA = max(a[1], b[1])
            let xB = min(a[2], b[2]), yB = min(a[3], b[3])
            let interW = max(0, xB - xA), interH = max(0, yB - yA)
            let inter = interW * interH
            let areaA = (a[2]-a[0]) * (a[3]-a[1])
            let areaB = (b[2]-b[0]) * (b[3]-b[1])
            return inter / (areaA + areaB - inter)
        }
        
        static func maskToPolygons(
            m: [UInt8],
            width: Int,
            h: Int) throws -> [[CGPoint]] {
                var mask = m
                var height = h
                // 1️⃣ Binary data ➜ one‑channel CGImage ------------------------------
                if width * height != mask.count {
                    print("NOT CORRECT BOUNDS")
                    return []
                }
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
                
                print("mask image size for poly: \(width)x\(height) \(cgMask.width)x\(cgMask.height)")
                
                // 2️⃣ Vision contour detection ---------------------------------------
                let request = VNDetectContoursRequest()
                request.detectsDarkOnLight   = false   // mask = white foreground
                request.contrastAdjustment   = 1.0     // no stretching needed
                request.maximumImageDimension = max(width, height)  // keep full res
                
                try VNImageRequestHandler(cgImage: cgMask, options: [:])
                    .perform([request])
                
                guard let obs = request.results?.first as? VNContoursObservation else {
                    print("no polygons found")
                    return []
                }
                // 3️⃣ Copy contours + denormalise y‑axis (Vision y: 0=bottom, UIKit 0=top)
                var polygons = [[CGPoint]]()
                print("\(obs.topLevelContourCount) polygons found")
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
}

extension Array {
    /// Splits the array into equally sized sub-arrays.
    /// The last chunk is the remainder (like Python’s list slicing).
    func chunked(into size: Int) -> [[Element]] {
        stride(from: 0, to: count, by: size).map {
            Array(self[$0 ..< Swift.min($0 + size, count)])
        }
    }
}

@available(iOS 18.0, *)
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

@available(iOS 18.0, *)
func tensorSum(_ tensor: MLTensor) throws -> Float {
    // 1️⃣ reduce across all axes (keeps scalar rank‑0 tensor)
    let s = tensor.sum()                     // element‑wise sum ‑> [ ]

    // 2️⃣ materialise that scalar as a shaped array
    let scalarSA: MLShapedArray<Float> = try shapedArraySync(s)

    // 3️⃣ shaped array of rank‑0 stores its value at .scalars[0]
    return scalarSA.scalars[0]
}
