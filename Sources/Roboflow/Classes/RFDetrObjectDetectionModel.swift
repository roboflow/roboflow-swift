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
@available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
public class RFDetrObjectDetectionModel: RFObjectDetectionModel {

    public override init() {
        super.init()
    }
    
    /// Load a local RFDetr model file
    @available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
    public func loadLocalRFDetrModel(modelPath: URL, colors: [String: String], classes: [String]) -> Error? {
        self.colors = colors
        self.classes = classes
        do {
            let config = MLModelConfiguration()
            var modelURL = modelPath
            
            // If the model is .mlpackage, compile it first
            if modelPath.pathExtension == "mlpackage" {
                do {
                    let compiledModelURL = try MLModel.compileModel(at: modelPath)
                    modelURL = compiledModelURL
                } catch {
                    return error
                }
            }
            
            mlModel = try RFDetr(contentsOf: modelURL, configuration: config).model
            
            // Note: RFDetr models don't use VNCoreMLModel for post-processing
            // We'll handle the raw output directly
            
        } catch {
            return error
        }
        return nil
    }
    
    /// Load the retrieved CoreML model for RFDetr
    override func loadMLModel(modelPath: URL, colors: [String: String], classes: [String]) -> Error? {
        return loadLocalRFDetrModel(modelPath: modelPath, colors: colors, classes: classes)
    }
    
    /// Run image through RFDetr model and return object detection predictions
    @available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
    public override func detect(pixelBuffer buffer: CVPixelBuffer, completion: @escaping (([RFPrediction]?, Error?) -> Void)) {
        guard let mlModel = self.mlModel else {
            completion(nil, NSError(domain: "RFDetrObjectDetectionModel", code: 1, userInfo: [NSLocalizedDescriptionKey: "Model initialization failed."]))
            return
        }
        
        do {
            // Create RFDetr input from pixel buffer
            let input = try RFDetrInput(pixelBuffer: buffer)
            
            // Run prediction with RFDetr
            let rfdetr = RFDetr(model: mlModel)
            let output = try rfdetr.prediction(input: input)
            
            // Process RFDetr outputs to create detection objects
            let detections = try processRFDetrOutputs(
                boxes: output.boxes,
                scores: output.scores,
                labels: output.labels,
                imageWidth: Int(buffer.width()),
                imageHeight: Int(buffer.height())
            )
            
            completion(detections, nil)
        } catch let error {
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
    
    // Store class list for processing
    private var classes: [String] = []
} 