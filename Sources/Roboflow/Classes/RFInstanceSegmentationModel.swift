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

//Creates an instance of an ML model that's hosted on Roboflow
public class RFInstanceSegmentationModel: RFObjectDetectionModel {
    var classes = [String]()
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
        guard let coreMLRequest = self.coreMLRequest else {
            completion(nil, "Model initialization failed.")
            return
        }
        guard let ciImage = CIImage(image: image) else {
            completion(nil, "Image failed.")
            return
        }
        let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])

        do {
            try handler.perform([coreMLRequest])
            
            guard let detectResults = coreMLRequest.results else { return }
            
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
            let protoPtr = proto.dataPointer.bindMemory(to: Float.self,
                                                        capacity: proto.count)
            let preds = UnsafeBufferPointer(start: p, count: pred.count)
            let protoShape = (c:Int(truncating: proto.shape[1]),
                              h:Int(truncating: proto.shape[2]),
                              w:Int(truncating: proto.shape[3]))
            
            // each row = 5 + numCls + numMasks
            let stride = 4 + numCls + numMasks
            let numDet = preds.count / stride
            
            var detRows = [[Float]]()
            var coeffs  = [[Float]]()
            var detections = [RFObjectDetectionPrediction]()
            
                    
            for i in 0..<numDet {
                var coords = [Float]()
                for j in 0..<4 {
                    coords.append(preds[j*numDet + i])
                }
                
                var confidences = [Float]()
                for j in 4..<4+numCls {
                    confidences.append(preds[j*numDet + i])
                }
                
                var coeff = [Float]()
                for j in 4+numCls..<stride {
                    coeff.append(preds[j*numDet + i])
                }
                
                // class confs
                guard let (maxConf,classID) = confidences.enumerated()
                          .map({ ($0.element,$0.offset) })
                          .max(by: {$0.0 < $1.0}),
                      maxConf >= Float(self.threshold) else { continue }
                
                
                // xywh → xyxy
                let cx = coords[0], cy = coords[1]
                let w  = coords[2], h = coords[3]
                let x1 = cx - w/2, y1 = cy - h/2
                let x2 = cx + w/2, y2 = cy + h/2
                
                var row = [x1,y1,x2,y2, maxConf]
                row.append(Float(classID))
                coeffs.append(coeff)
                detRows.append(row)
            }
            
            
            let kept = nonMaxSuppressionFast(detRows, iouThresh: Float(self.overlap))
            var final: [RFObjectDetectionPrediction] = []
            
            var boxesKeep = [CGRect]()
            var coeffsKeep = [[Float]]()
            var keeps = [[Float]]()
            
            // gather masks only for kept indices
            for keep in kept {
                guard let idx = detRows.firstIndex(where: { $0.elementsEqual(keep) }) else { continue }
                
                let x = (keep[0] + keep[2])/2
                let y = (keep[1] + keep[3])/2
                let width = (keep[2] - keep[0])
                let height = (keep[3] - keep[1])
                let minX = CGFloat(keep[0])
                let minY = 640 - CGFloat(keep[1] + height)
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
                                                             imgH: Int(image.size.height), imgW: Int(image.size.width))
                
                for (i, maskBin) in maskBins.enumerated() {
                    let h = maskBin.count                 // rows
                    let w = maskBin.first!.count          // cols
                    let flatMask: [UInt8] = maskBin.flatMap { $0 }
                    
                    let polys = try? MaskUtils.maskToPolygons(mask: flatMask, width: w, height: h)
                    
                    var polygon = [CGPoint]()
                    if let polygons = polys {
                        if polygons.count > 0 {
                            polygon = polygons[0]
                            if (polygon.count > 0) {
                                polygon.append(polygon.first!)
                            }
                        }
                    }
                    
                    let keep = keeps[i]
                    let box = boxesKeep[i]
                    
                    let classname = classes[Int(keep[5])]
                    let detection = RFInstanceSegmentationPrediction(x: Float(box.midX), y: Float(h) - Float(box.midY), width: Float(box.width), height: Float(box.height), className: classname, confidence: keep[4], color: hexStringToUIColor(hex: colors[classname] ?? "#ff0000"), box: box, points:polygon, mask: maskBin)
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
        ) -> MLTensor {                    // one crop per detection
            let (c, mh, mw) = protoShape
            let shapedProto = MLShapedArray<Float>(proto)
            var masks = MLTensor(shapedProto)
            masks = masks.reshaped(to: [c, mh * mw])
            
            let rows = coeffs.count
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
            let coeffs_t = MLTensor(shapedCoeffs)
            masks = coeffs_t.matmul(masks)
            masks = ((masks * -1)
                .exp() + 1)
                .reciprocal() // sigmoid
            
            masks = masks.reshaped(to: [rows, mh, mw])

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
                                        imgW: Int) -> [[[UInt8]]] {
            let rawMasks = preprocessSegmentationMasks(proto: proto,
                                                       protoShape: protoShape,
                                                       coeffs: coeffs,
                                                       outH: imgH, outW: imgW)
            let masks = rawMasks.resized(to: (imgW, imgH), method: .bilinear(alignCorners: false))
            
            let (h, w) = (masks.shape[1], masks.shape[2])
            let nDet = dets.count

            let rowVector = (0..<w).map { Float($0) }
            let colVector = (0..<h).map { Float($0) }

            let r = MLTensor(MLShapedArray<Float>(scalars: rowVector, shape: [w]))
                        .reshaped(to: [1, 1, w])

            let c = MLTensor(MLShapedArray<Float>(scalars: colVector, shape: [h]))
                        .reshaped(to: [1, h, 1])
            var result = [[[UInt8]]]()
            for i in 0..<nDet {
                // get detection
                let det = dets[i]
                let x1 = Float(det.minX)
                let y1 = Float(det.minY)
                let x2 = Float(det.maxX)
                let y2 = Float(det.maxY)
                
                let rotatedMask = masks[i]
                
                let kx = (r .> (x1))
                
                let keep = kx .&     // r >= x1
                (r .< x2) .&    // & r < x2
                (c .> (Float(h)-y2)) .&
                (c .< (Float(h)-y1 ))   // final shape [nDet,h,w]
                
                let cropped = rotatedMask * keep
                
                let shaped: MLShapedArray<Float> = try! shapedArraySync(cropped, as: Float.self)
                

                
                var out: [[UInt8]] =
                                                 Array(repeating:
                                                       Array(repeating: 0, count: w),
                                                       count: h)
                let scalars = shaped.scalars                      // flat Float buffer
                    var offset  = 0
                        for y in 0..<h {
                            for x in 0..<w {
                                let p = scalars[offset]
                                out[y][x] = p < 0.5 ? 0 : 255
                                offset += 1
                            }
                        }
                result.append(out)
                
            }
            return result
        }
        
        static func maskToPolygons(
            mask: [UInt8],
            width: Int,
            height: Int) throws -> [[CGPoint]] {

            // 1️⃣ Binary data ➜ one‑channel CGImage ------------------------------
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

    // MARK: – Greedy NMS (non_max_suppression_fast) ------------------------------

    func nonMaxSuppressionFast(_ dets: [[Float]], iouThresh: Float) -> [[Float]] {
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

    func iouRectIoU(_ a:[Float], _ b:[Float]) -> Float {
        let xA = max(a[0], b[0]), yA = max(a[1], b[1])
        let xB = min(a[2], b[2]), yB = min(a[3], b[3])
        let interW = max(0, xB - xA), interH = max(0, yB - yA)
        let inter = interW * interH
        let areaA = (a[2]-a[0]) * (a[3]-a[1])
        let areaB = (b[2]-b[0]) * (b[3]-b[1])
        return inter / (areaA + areaB - inter)
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

    Task.detached {
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
