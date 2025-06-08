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

//Creates an instance of an ML model that's hosted on Roboflow
public class RFInstanceSegmentationModel: RFObjectDetectionModel {
    var classes = [String]()
    var maskProcessingMode: MaskProcessingMode = .balanced
    //Load the retrieved CoreML model into an already created RFObjectDetectionModel instance
    override func loadMLModel(modelPath: URL, colors: [String: String], classes: [String]) -> Error? {
        self.colors = colors
        self.classes = classes
        do {
            let config = MLModelConfiguration()
            if #available(iOS 16.0, *) {
                config.computeUnits = .cpuAndNeuralEngine
            } else {
                // Fallback on earlier versions
            }
            mlModel = try yolov8_seg(contentsOf: modelPath, configuration: config).model
            visionModel = try VNCoreMLModel(for: mlModel)
            visionModel.featureProvider = super.thresholdProvider
            let request = VNCoreMLRequest(model: visionModel)
            request.imageCropAndScaleOption = .scaleFill
            coreMLRequest = request
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
            try handler.perform([coreMLRequest])
            guard let detectResults = coreMLRequest.results else { return }
            
            let predictions = detectResults[1] as! VNCoreMLFeatureValueObservation
            let protos = detectResults[0] as! VNCoreMLFeatureValueObservation
            
            let pred = predictions.featureValue.multiArrayValue!
            let proto = protos.featureValue.multiArrayValue!
            
            
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
            
            if #available(iOS 18.0, *) {
                let maskBins = MaskUtils.processMaskAccurate(proto: proto,
                                                             protoShape: protoShape,
                                                             coeffs: coeffsKeep,
                                                             dets: boxesKeep,
                                                             imgH: Int(image.size.height), imgW: Int(image.size.width),
                                                             procH: Int(outputSize.height), procW: Int(outputSize.width))
                
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
                    var polys = [[CGPoint]]()
                    do {
                        polys = try MaskUtils.maskToPolygons(m: flatMask, width: roiW, h: roiH)
                    } catch {
                        print(error)
                    }
                    
                    var polygon = (polys.count > 0 ? polys[0] : [])
                    
                    
                    polygon = polygon.map { CGPoint(x: CGFloat($0.x + box.minX), y: CGFloat($0.y + box.minY)) }
                    
                    let keep = keeps[i]
                    
                    let classname = classes[Int(keep[5])]
                    let detection = RFInstanceSegmentationPrediction(x: Float(box.midX), y: Float(box.midY), width: Float(box.width), height: Float(box.height), className: classname, confidence: keep[4], color: hexStringToUIColor(hex: colors[classname] ?? "#ff0000"), box: box, points:polygon, mask: nil)
                    final.append(detection)
                }
            }
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
}
