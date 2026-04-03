//
//  RFYoloLiteObjectDetectionModel.swift
//  Roboflow
//
//  Created by AI Assistant
//

import Foundation
import CoreML
import Vision

/// Object detection model that uses YOLOLite for inference.
///
/// The CoreML model includes baked-in NMS, so VNCoreML returns
/// VNRecognizedObjectObservation results directly — same path as YOLOv5.
/// Runtime thresholds (iouThreshold, confidenceThreshold) are adjustable
/// via the inherited ThresholdProvider.
///
/// Uses `.cpuAndGPU` compute units to avoid Neural Engine fp16 underflow.
public class RFYoloLiteObjectDetectionModel: RFObjectDetectionModel {

    /// Load the retrieved CoreML model for YOLOLite
    override func loadMLModel(modelPath: URL, colors: [String: String], classes: [String], environment: [String: Any]) -> Error? {
        self.colors = colors
        self.classes = classes
        self.environment = environment
        self.modelPath = modelPath

        do {
            if #available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *) {
                let config = MLModelConfiguration()
                // Use CPU+GPU only — the Neural Engine ignores compute_precision=FLOAT32
                // and runs in fp16, which causes underflow in the decode head.
                config.computeUnits = .cpuAndGPU

                mlModel = try MLModel(contentsOf: modelPath, configuration: config)

                do {
                    visionModel = try VNCoreMLModel(for: mlModel)
                    visionModel.featureProvider = thresholdProvider
                    let request = VNCoreMLRequest(model: visionModel)
                    request.imageCropAndScaleOption = .scaleFill
                    coreMLRequest = request
                } catch {
                    print("Error to initialize YOLOLite model: \(error)")
                }
            } else {
                return UnsupportedOSError()
            }
        } catch {
            return error
        }
        return nil
    }

    /// Run image through YOLOLite model and return object detection predictions.
    ///
    /// CoreML NMS returns all N entries with suppressed ones zeroed.
    /// We filter to non-zero observations and use the per-class label
    /// confidence (in [0, 1]) rather than the observation-level confidence
    /// which VNCoreML computes as a sum of per-class probabilities.
    public override func detect(pixelBuffer buffer: CVPixelBuffer, completion: @escaping (([RFPrediction]?, Error?) -> Void)) {
        guard let coreMLRequest = self.coreMLRequest else {
            completion(nil, NSError(domain: "RFYoloLiteObjectDetectionModel", code: 1,
                                    userInfo: [NSLocalizedDescriptionKey: "VNCoreML model initialization failed."]))
            return
        }

        let handler = VNImageRequestHandler(cvPixelBuffer: buffer)
        do {
            try handler.perform([coreMLRequest])

            guard let detectResults = coreMLRequest.results as? [VNRecognizedObjectObservation] else {
                completion([], nil)
                return
            }

            var detections: [RFObjectDetectionPrediction] = []
            for result in detectResults {
                guard result.confidence > 0 else { continue }

                let flippedBox = CGRect(x: result.boundingBox.minX,
                                        y: 1 - result.boundingBox.maxY,
                                        width: result.boundingBox.width,
                                        height: result.boundingBox.height)
                let box = VNImageRectForNormalizedRect(flippedBox, Int(buffer.width()), Int(buffer.height()))

                var label = ""
                var confidence: Float = result.confidence

                if let topLabel = result.labels.first {
                    label = topLabel.identifier
                    // Use per-class confidence (0-1) instead of observation-level (can be >1)
                    confidence = topLabel.confidence

                    if let intValue = Int(label), !classes.contains(label), intValue < classes.count {
                        label = classes[intValue]
                    }
                }

                detections.append(RFObjectDetectionPrediction(
                    x: Float((box.maxX + box.minX) / 2.0),
                    y: Float((box.maxY + box.minY) / 2.0),
                    width: Float(box.maxX - box.minX),
                    height: Float(box.maxY - box.minY),
                    className: label,
                    confidence: confidence,
                    color: hexStringToCGColor(hex: colors[label] ?? "#ff0000"),
                    box: box
                ))
            }
            completion(detections, nil)
        } catch {
            completion(nil, error)
        }
    }
}
