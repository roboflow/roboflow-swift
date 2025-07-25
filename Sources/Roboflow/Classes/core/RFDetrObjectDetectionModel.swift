//
//  RFDetrObjectDetectionModel.swift
//  Roboflow
//
//  Created by AI Assistant
//

import Foundation
import CoreML
import Vision

/// Object detection model that uses RFDetr for inference
public class RFDetrObjectDetectionModel: RFObjectDetectionModel {

    public override init() {
        super.init()
    }
    
    /// Load the retrieved CoreML model for RFDetr
    override func loadMLModel(modelPath: URL, colors: [String: String], classes: [String], environment: [String: Any]) -> Error? {
        self.colors = colors
        self.classes = classes
        self.environment = environment
        self.modelPath = modelPath
        
        do {
            if #available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *) {
                let config = MLModelConfiguration()
                if #available(iOS 16.0, *) {
                    config.computeUnits = .cpuAndNeuralEngine
                } else {
                    // Fallback on earlier versions
                }
                
                // Use CPU-only execution on iOS Simulator to avoid Metal compatibility issues
                #if targetEnvironment(simulator)
                    config.computeUnits = .cpuOnly
                #endif
                
                // Load the compiled model directly (modelPath should point to .mlmodelc)
                mlModel = try RFDetr(contentsOf: modelPath, configuration: config).model
                
                // Try to create VNCoreMLModel for macOS 10.15+ / iOS 13.0+
                do {
                    visionModel = try VNCoreMLModel(for: mlModel)
                    let request = VNCoreMLRequest(model: visionModel)
                    request.imageCropAndScaleOption = .scaleFill
                    coreMLRequest = request
                } catch {
                    print("Error to initialize RFDetr model: \(error)")
                }
            } else {
                return UnsupportedOSError()
            }
        } catch {
            return error
        }
        return nil
    }

    /// Run image through RFDetr model and return object detection predictions
    public override func detect(pixelBuffer buffer: CVPixelBuffer, completion: @escaping (([RFPrediction]?, Error?) -> Void)) {
        
        // Try VNCoreML approach first (macOS 10.15+ / iOS 13.0+)
        if #available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *) {
            detectWithVNCoreML(pixelBuffer: buffer, completion: completion)
        } else {
            // Fallback to direct MLModel usage for earlier versions
            completion(nil, UnsupportedOSError())
        }
    }
    
    /// VNCoreML-based detection for macOS 10.15+ / iOS 13.0+
    @available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
    private func detectWithVNCoreML(pixelBuffer buffer: CVPixelBuffer, completion: @escaping (([RFPrediction]?, Error?) -> Void)) {
        guard let coreMLRequest = self.coreMLRequest else {
            completion(nil, NSError(domain: "RFDetrObjectDetectionModel", code: 1, userInfo: [NSLocalizedDescriptionKey: "VNCoreML model initialization failed."]))
            return
        }
        
        let handler = VNImageRequestHandler(cvPixelBuffer: buffer)
        
        do {
            try handler.perform([coreMLRequest])
            
            // For RFDetr models, we need to access the raw MLFeatureProvider results
            // since they don't return standard VNDetectedObjectObservation objects
            guard let results = coreMLRequest.results as? [VNCoreMLFeatureValueObservation] else {
                completion(nil, NSError(domain: "RFDetrObjectDetectionModel", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to get RFDetr model outputs"]))
                return
            }
            
            // Extract the raw outputs from the VNCoreMLFeatureValueObservation
            var boxes: MLMultiArray?
            var scores: MLMultiArray?
            var labels: MLMultiArray?
            
            for result in results {
                switch result.featureName {
                case "boxes":
                    boxes = result.featureValue.multiArrayValue
                case "scores":
                    scores = result.featureValue.multiArrayValue
                case "labels":
                    labels = result.featureValue.multiArrayValue
                default:
                    break
                }
            }
            
            guard let boxesArray = boxes,
                let scoresArray = scores,
                let labelsArray = labels else {
                completion(nil, NSError(domain: "RFDetrObjectDetectionModel", code: 3, userInfo: [NSLocalizedDescriptionKey: "Missing required RFDetr outputs (boxes, scores, labels)"]))
                return
            }
            
            // Process RFDetr outputs to create detection objects
            let detections = try processRFDetrOutputs(
                boxes: boxesArray,
                scores: scoresArray,
                labels: labelsArray,
                imageWidth: Int(buffer.width()),
                imageHeight: Int(buffer.height())
            )
            completion(detections, nil)            
        } catch {
            completion(nil, error)
        }
    }
    
    /// Process RFDetr raw outputs into RFObjectDetectionPrediction objects
    private func processRFDetrOutputs(boxes: MLMultiArray, scores: MLMultiArray, labels: MLMultiArray, imageWidth: Int, imageHeight: Int) throws -> [RFObjectDetectionPrediction] {
        var detections: [RFObjectDetectionPrediction] = []
        
        // Get array dimensions - RFDetr outputs are [1, 300] for scores/labels and [1, 300, 4] for boxes
        let numDetections = scores.shape[1].intValue // 300 detections
        
        // Process each detection
        for i in 0..<numDetections {
            // Get confidence score (RFDetr format: [batch, detection_index])
            let confidence = Float(scores[[0, i] as [NSNumber]].doubleValue)
            
            // Skip detections below threshold
            if confidence < Float(threshold) {
                continue
            }
            
            // Get class label (RFDetr format: [batch, detection_index])
            let classIndex = Int(labels[[0, i] as [NSNumber]].int32Value)
            let className = classIndex < classes.count ? classes[classIndex] : "unknown"
            
            // Get bounding box coordinates (RFDetr format: [batch, detection_index, coordinate])
            // RFDetr typically outputs [center_x, center_y, width, height] in normalized coordinates
            let centerX_norm = Float(boxes[[0, i, 0] as [NSNumber]].doubleValue)
            let centerY_norm = Float(boxes[[0, i, 1] as [NSNumber]].doubleValue) 
            let width_norm = Float(boxes[[0, i, 2] as [NSNumber]].doubleValue)
            let height_norm = Float(boxes[[0, i, 3] as [NSNumber]].doubleValue)
            
            // Convert normalized coordinates to pixel coordinates
            let centerX = centerX_norm * Float(imageWidth)
            let centerY = centerY_norm * Float(imageHeight)
            let width = abs(width_norm) * Float(imageWidth)  // Use abs() to handle negative values
            let height = abs(height_norm) * Float(imageHeight)
            
            // Skip invalid boxes
            if width <= 0 || height <= 0 {
                continue
            }
            
            // Convert center coordinates to top-left corner for CGRect
            let x1 = centerX - width / 2.0
            let y1 = centerY - height / 2.0
            
            // Create bounding box rect
            let box = CGRect(x: CGFloat(x1), y: CGFloat(y1), width: CGFloat(width), height: CGFloat(height))
            
            // Get color for this class
            let color = hexStringToCGColor(hex: colors[className] ?? "#ff0000")
            
            // Create detection object
            let detection = RFObjectDetectionPrediction(
                x: centerX,
                y: centerY,
                width: width,
                height: height,
                className: className,
                confidence: confidence,
                color: color,
                box: box
            )
            
            detections.append(detection)
        }
        
        return detections
    }
} 
