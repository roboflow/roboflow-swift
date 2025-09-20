//
//  RFObjectDetectionModel.swift
//  Roboflow
//
//  Created by Nicholas Arner on 4/12/22.
//

import Foundation
import CoreML
import Vision
import Accelerate



//Creates an instance of an ML model that's hosted on Roboflow
public class RFInstanceSegmentationModel: RFObjectDetectionModel {
    
    
    //Load the retrieved CoreML model into an already created RFObjectDetectionModel instance
    override func loadMLModel(modelPath: URL, colors: [String: String], classes: [String], environment: [String: Any]) -> Error? {
        let _ = super.loadMLModel(modelPath: modelPath, colors: colors, classes: classes, environment: environment)
        do {
            
            if #available(iOS 16.0, macOS 13.0, *) {
                let config = MLModelConfiguration()
                config.computeUnits = .cpuAndNeuralEngine
                mlModel = try yolov8_seg(contentsOf: modelPath, configuration: config).model
                visionModel = try VNCoreMLModel(for: mlModel)
                visionModel.featureProvider = super.thresholdProvider
                let request = VNCoreMLRequest(model: visionModel)
                request.imageCropAndScaleOption = .scaleFill
                coreMLRequest = request
            } else {
                // Fallback on earlier versions
                return UnsupportedOSError()
            }
           
        } catch {
            return error
        }
        return nil
    }
    
    //Run image through model and return Detections
    public override func detect(pixelBuffer:CVPixelBuffer, completion: @escaping (([RFPrediction]?, Error?) -> Void)) {
        let imgHeight = CGFloat(pixelBuffer.height())
        let imgWidth = CGFloat(pixelBuffer.width())
        
        let outputSize = self.maskProcessingMode == .performance ? CGSize(width: imgWidth / 2, height: imgHeight / 2) : self.maskProcessingMode == .balanced ? CGSize(width: 640, height: 640) : CGSize(width: imgWidth, height: imgHeight)
        guard let coreMLRequest = self.coreMLRequest else {
            completion(nil, "Model initialization failed.")
            return
        }
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])

        do {
            try handler.perform([coreMLRequest])
            guard let detectResults = coreMLRequest.results else { return }

            let castDetectResults0 = (detectResults[0] as! VNCoreMLFeatureValueObservation).featureValue.multiArrayValue!
            let castDetectResults1 = (detectResults[1] as! VNCoreMLFeatureValueObservation).featureValue.multiArrayValue!
            
            let pred = castDetectResults0.shape.count == 3 ? castDetectResults0 : castDetectResults1
            let proto = castDetectResults1.shape.count == 4 ? castDetectResults1 : castDetectResults0
            
            let numMasks = 32
            let numCls = self.colors.count
            
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
                    
            // MARK: -- constants and helpers
            let spatial   = numDet                         // number of rows in a “column block”

            // pre-allocate results to avoid repeated realloc
            detRows.reserveCapacity(pred.shape[2].intValue)
            coeffs .reserveCapacity(pred.shape[2].intValue)

            // flatten MLMultiArray to raw Float pointer
            let basePtr = pred.dataPointer.assumingMemoryBound(to: Float.self)

            // lock for the result buffers (used only when we keep a detection)
            let outLock = NSLock()
            
            let xs = outputSize.width / 640
            let ys = outputSize.height / 640
            
            // ---- read bbox (cx,cy,w,h)  ----
            @inline(__always) func col(_ k: Int, _ i: Int) -> Float {
                basePtr[k * spatial + i]       // fast pointer math, no multiply in inner loops
            }
            // MARK: -- parallel pass over detections
            DispatchQueue.concurrentPerform(iterations: numDet) { i in
                
                let cx = col(0, i), cy = col(1, i)
                let w  = col(2, i), h  = col(3, i)

                // ---- arg-max over class scores ----
                var bestScore: Float = 0
                var bestCls  : Int   = -1
                var k = 4                         // first class score column
                while k < 4 + numCls {
                    let s = col(k, i)
                    if s > bestScore { bestScore = s; bestCls = k-4 }
                    k &+= 1
                }
                guard bestScore >= Float(threshold) else { return }   // prunes most rows quickly

                // ---- collect mask coefficients ----
                var localCoeff = [Float](repeating: 0, count: numMasks)
                var cidx = 4 + numCls               // first coeff column
                for m in 0..<numMasks {
                    localCoeff[m] = col(cidx, i)
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
                detRows.append([bbox.x * Float(xs), bbox.y * Float(ys), bbox.z * Float(xs), bbox.w * Float(ys), bestScore, Float(bestCls)])
                coeffs .append(localCoeff)
                outLock.unlock()
            }
            
            var kept: [[Float]] = []
            if #available(iOS 18.0, macOS 15.0, *) {
                kept = MaskUtils.nonMaxSuppressionFast(detRows, iouThresh: Float(self.overlap))
            } else {
                // Fallback on earlier versions
                completion(nil, UnsupportedOSError())
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
                let box = CGRect(x: CGFloat(minX), y: CGFloat(minY), width: CGFloat(width), height: CGFloat(height))
                
                boxesKeep.append(box)
                coeffsKeep.append(coeffs[idx])
                keeps.append(keep)
                
            }
            
            if #available(iOS 18.0, macOS 15.0, *) {
                let maskBins = MaskUtils.processMaskAccurate(proto: proto,
                                                             protoShape: protoShape,
                                                             coeffs: coeffsKeep,
                                                             dets: boxesKeep,
                                                             procH: Int(outputSize.height), procW: Int(outputSize.width))
                
                let scaleX = Float(imgWidth / outputSize.width)
                let scaleY = Float(imgHeight / outputSize.height)
                for (i, maskBin) in maskBins.enumerated() {
                    let flatMask: [UInt8] = maskBin     // cols
                    let box = boxesKeep[i]
                    
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
                    var polys = [[CGPoint]]()
                    do {
                        polys = try MaskUtils.maskToPolygons(m: flatMask, width: roiW, h: roiH)
                    } catch {
                        print(error)
                    }
                    
                    // Select largest contour: If multiple contours exist, picks the one with the most points
                    // using max(by:) to match Python's argmax behavior
                    var polygon: [CGPoint] = []
                    if !polys.isEmpty {
                        polygon = polys.max(by: { $0.count < $1.count }) ?? []
                    }
                    
                    if polygon.count > maskMaxNumberPoints {
                        // ── 1. Decide which vertices to drop ──────────────────────────
                        let toKeep    = maskMaxNumberPoints
                        let strideF   = Double(polygon.count) / Double(toKeep)   // ≈ spacing
                        var keep = [CGPoint]()
                        
                        var cursor = strideF / 2            // centre the first drop
                        for _ in 0..<toKeep {
                            keep.append(polygon[Int(cursor.rounded()) % polygon.count])
                            cursor += strideF
                        }
                        
                        polygon = keep
                    }
                    
                    polygon = polygon.map { CGPoint(x: CGFloat($0.x + box.minX) * CGFloat(scaleX), y: CGFloat($0.y + box.minY) * CGFloat(scaleY)) }
                    
                    let keep = keeps[i]
                    
                    let classname = classes[Int(keep[5])]
                    let detection = RFInstanceSegmentationPrediction(x: Float(box.midX) * scaleX, y: Float(box.midY) * scaleY, width: Float(box.width) * scaleX, height: Float(box.height) * scaleY, className: classname, confidence: keep[4], color: hexStringToCGColor(hex: colors[classname] ?? "#ff0000"), box: box, points:polygon)
                    final.append(detection)
                }
            } else {
                completion(nil, UnsupportedOSError())
            }
            completion(final, nil)
        } catch let error {
            completion(nil, error)
        }
    }
}
