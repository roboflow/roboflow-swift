//
//  RFDetrInstanceSegmentationModel.swift
//  Roboflow
//
//  Instance segmentation for RF-DETR CoreML exports.
//
//  RF-DETR seg exports emit four named outputs (see rfdetr_coreml.py):
//    boxes  [1, K, 4]      normalized cxcywh
//    scores [1, K]         confidence (sigmoid already applied via top-k)
//    labels [1, K]         class index
//    masks  [1, K, Hm, Wm] raw mask logits at resolution / mask_downsample_ratio
//
//  This differs entirely from the YOLOv8-seg layout handled by
//  RFInstanceSegmentationModel, so it gets its own decoder. Box/score/label
//  parsing mirrors RFDetrObjectDetectionModel; we additionally sigmoid each
//  per-query mask, threshold it, and trace a polygon contour.
//

import Foundation
import CoreML
import Vision

public class RFDetrInstanceSegmentationModel: RFDetrObjectDetectionModel {

    public override init() {
        super.init()
    }

    // Loading is identical to RF-DETR detection (same RFDetr wrapper, same
    // VNCoreMLRequest setup) — the extra `masks` output is just read at detect time.

    public override func detect(pixelBuffer buffer: CVPixelBuffer, completion: @escaping (([RFPrediction]?, Error?) -> Void)) {
        guard #available(macOS 15.0, iOS 18.0, *) else {
            completion(nil, UnsupportedOSError())
            return
        }
        guard let coreMLRequest = self.coreMLRequest else {
            completion(nil, NSError(domain: "RFDetrInstanceSegmentationModel", code: 1,
                                    userInfo: [NSLocalizedDescriptionKey: "VNCoreML model initialization failed."]))
            return
        }

        let handler = VNImageRequestHandler(cvPixelBuffer: buffer)
        do {
            try handler.perform([coreMLRequest])

            guard let results = coreMLRequest.results as? [VNCoreMLFeatureValueObservation] else {
                completion(nil, NSError(domain: "RFDetrInstanceSegmentationModel", code: 2,
                                        userInfo: [NSLocalizedDescriptionKey: "Failed to get RF-DETR seg model outputs"]))
                return
            }

            var boxes: MLMultiArray?
            var scores: MLMultiArray?
            var labels: MLMultiArray?
            var masks: MLMultiArray?
            for result in results {
                switch result.featureName {
                case "boxes":  boxes  = result.featureValue.multiArrayValue
                case "scores": scores = result.featureValue.multiArrayValue
                case "labels": labels = result.featureValue.multiArrayValue
                case "masks":  masks  = result.featureValue.multiArrayValue
                default:       break
                }
            }

            guard let boxesArray = boxes, let scoresArray = scores,
                  let labelsArray = labels, let masksArray = masks else {
                completion(nil, NSError(domain: "RFDetrInstanceSegmentationModel", code: 3,
                                        userInfo: [NSLocalizedDescriptionKey: "Missing required RF-DETR seg outputs (boxes, scores, labels, masks)"]))
                return
            }

            let detections = try processOutputs(boxes: boxesArray, scores: scoresArray,
                                                 labels: labelsArray, masks: masksArray,
                                                 imageWidth: Int(buffer.width()),
                                                 imageHeight: Int(buffer.height()))
            completion(detections, nil)
        } catch {
            completion(nil, error)
        }
    }

    private func processOutputs(boxes: MLMultiArray, scores: MLMultiArray, labels: MLMultiArray,
                                masks: MLMultiArray, imageWidth: Int, imageHeight: Int) throws -> [RFInstanceSegmentationPrediction] {
        let numDet = scores.shape[1].intValue
        let maskH  = masks.shape[2].intValue   // Hm
        let maskW  = masks.shape[3].intValue   // Wm
        // Each mask covers the full image, so map mask-grid pixels straight to image pixels.
        let sx = Float(imageWidth)  / Float(maskW)
        let sy = Float(imageHeight) / Float(maskH)

        var detections: [RFInstanceSegmentationPrediction] = []

        for i in 0..<numDet {
            let confidence = Float(scores[[0, i] as [NSNumber]].doubleValue)
            if confidence < Float(threshold) { continue }

            let classIndex = Int(labels[[0, i] as [NSNumber]].int32Value)
            let className = classIndex < classes.count ? classes[classIndex] : "unknown"

            // Box: normalized cxcywh -> pixel xywh
            let cx = Float(boxes[[0, i, 0] as [NSNumber]].doubleValue) * Float(imageWidth)
            let cy = Float(boxes[[0, i, 1] as [NSNumber]].doubleValue) * Float(imageHeight)
            let w  = abs(Float(boxes[[0, i, 2] as [NSNumber]].doubleValue)) * Float(imageWidth)
            let h  = abs(Float(boxes[[0, i, 3] as [NSNumber]].doubleValue)) * Float(imageHeight)
            if w <= 0 || h <= 0 { continue }

            let box = CGRect(x: CGFloat(cx - w / 2), y: CGFloat(cy - h / 2),
                             width: CGFloat(w), height: CGFloat(h))

            // Mask: sigmoid(logits) > 0.5 -> binary 0/255 grid, then trace polygon.
            var bin = [UInt8](repeating: 0, count: maskH * maskW)
            for r in 0..<maskH {
                for c in 0..<maskW {
                    let logit = Float(masks[[0, i, r, c] as [NSNumber]].doubleValue)
                    let prob = 1.0 / (1.0 + exp(-logit))
                    bin[r * maskW + c] = prob > 0.5 ? 255 : 0
                }
            }

            var polygon: [CGPoint] = []
            if #available(iOS 18.0, macOS 15.0, *) {
                let polys = (try? MaskUtils.maskToPolygons(m: bin, width: maskW, h: maskH)) ?? []
                polygon = polys.max(by: { $0.count < $1.count }) ?? []
            }

            // Downsample to maskMaxNumberPoints (same scheme as the YOLOv8-seg path).
            if polygon.count > maskMaxNumberPoints {
                let toKeep = maskMaxNumberPoints
                let strideF = Double(polygon.count) / Double(toKeep)
                var keep = [CGPoint]()
                var cursor = strideF / 2
                for _ in 0..<toKeep {
                    keep.append(polygon[Int(cursor.rounded()) % polygon.count])
                    cursor += strideF
                }
                polygon = keep
            }

            // Mask-grid space -> image pixel space.
            polygon = polygon.map { CGPoint(x: CGFloat(Float($0.x) * sx),
                                            y: CGFloat(Float($0.y) * sy)) }

            let color = hexStringToCGColor(hex: colors[className] ?? "#ff0000")
            detections.append(RFInstanceSegmentationPrediction(
                x: cx, y: cy, width: w, height: h,
                className: className, confidence: confidence,
                color: color, box: box, points: polygon))
        }

        return detections
    }
}