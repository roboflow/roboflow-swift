//
//  RFClassificationModel.swift
//  Roboflow
//
//  Created by Maxwell Stone on 6/16/25.
//

import Foundation
import CoreML
import Vision

//Creates an instance of an ML classification model
public class RFClassificationModel: RFModel {

    public override init() {
        super.init()
    }
    
    //Default model configuration parameters
    var multiclass: Bool = false
    
    //Load the retrieved CoreML model into an already created RFClassificationModel instance
    override func loadMLModel(modelPath: URL, colors: [String: String], classes: [String], environment: [String: Any]) -> Error? {
        let _ = super.loadMLModel(modelPath: modelPath, colors: colors, classes: classes, environment: environment)
        if let _ = environment["MULTICLASS"] {
            self.multiclass = true
        }
        do {
            if #available(macOS 10.14, *) {
                let config = MLModelConfiguration()
                if #available(macOS 10.15, *) {
                    mlModel = try MLModel(contentsOf: modelPath, configuration: config)
                } else {
                    // Fallback on earlier versions
                    return UnsupportedOSError()
                }
                visionModel = try VNCoreMLModel(for: mlModel)
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
    
    //Run image through model and return Classification predictions
    public override func detect(pixelBuffer buffer: CVPixelBuffer, completion: @escaping (([RFPrediction]?, Error?) -> Void)) {
        guard let mlModel = self.mlModel else {
            completion(nil, NSError(domain: "RFClassificationModel", code: 1, userInfo: [NSLocalizedDescriptionKey: "Model initialization failed."]))
            return
        }
        
        do {
            // Get the actual input name from model description
            guard let inputName = mlModel.modelDescription.inputDescriptionsByName.keys.first,
                let inputDescription = mlModel.modelDescription.inputDescriptionsByName[inputName] else {
                completion(nil, NSError(domain: "RFClassificationModel", code: 2, userInfo: [NSLocalizedDescriptionKey: "Could not find model input name"]))
                return
            }
            
            
            // Get expected image dimensions
            let inputImage: MLFeatureValue
            if let imageConstraint = inputDescription.imageConstraint {
                let targetWidth = Int(imageConstraint.pixelsWide)
                let targetHeight = Int(imageConstraint.pixelsHigh)
                
                // Create a resized pixel buffer
                let resizedBuffer = resizePixelBuffer(buffer, targetWidth: targetWidth, targetHeight: targetHeight)
                inputImage = MLFeatureValue(pixelBuffer: resizedBuffer ?? buffer)
            } else {
                // Use original buffer if no constraints
                inputImage = MLFeatureValue(pixelBuffer: buffer)
            }
            
            let inputDict = [inputName: inputImage]
            let inputProvider = try MLDictionaryFeatureProvider(dictionary: inputDict)
            
            // Run prediction directly with CoreML
            let prediction = try mlModel.prediction(from: inputProvider)
            
            // Get the actual output name from model description
            guard let outputName = mlModel.modelDescription.outputDescriptionsByName.keys.first,
                  let output = prediction.featureValue(for: outputName) else {
                completion(nil, NSError(domain: "RFClassificationModel", code: 3, userInfo: [NSLocalizedDescriptionKey: "Could not find model output"]))
                return
            }
            
            var predictions: [RFPrediction] = []
            
            // Handle different output types
            if let multiArray = output.multiArrayValue {
                // Multi-array output (may be logits that need softmax conversion)
                let rawValues = multiArray.dataPointer.bindMemory(to: Float.self, capacity: multiArray.count)
                
                // Check if values are logits (outside 0-1 range) and need softmax
                let probabilities: [Float]
                if !multiclass {
                    // Apply softmax to convert logits to probabilities
                    probabilities = applySoftmax(logits: rawValues, count: multiArray.count)
                } else {
                    // Values are already probabilities (sigmoid applied in model)
                    probabilities = Array(UnsafeBufferPointer(start: rawValues, count: multiArray.count))
                }
                
                for i in 0..<multiArray.count {
                    let confidence = probabilities[i]
                    if confidence >= Float(threshold) {
                        let prediction = RFClassificationPrediction(
                            className: self.classes[i], // Generic class name
                            confidence: confidence,
                            classId: i
                        )
                        predictions.append(prediction)
                    }
                }
            } else {
                // Unknown output type - log and continue with empty predictions
                print("Unknown output type: \(output.type)")
            }
            
            // Sort by confidence (highest first) - cast to RFClassificationPrediction since that's what we create
            predictions.sort { 
                guard let pred1 = $0 as? RFClassificationPrediction, 
                    let pred2 = $1 as? RFClassificationPrediction else { 
                    return false 
                }
                return pred1.confidence > pred2.confidence
            }
            
            if multiclass {
                completion(predictions, nil)
            } else {
                completion(predictions.isEmpty ? [] : [predictions[0]], nil)
            }
        } catch let error {
            completion(nil, error)
        }
    }
    
    // Helper function to resize CVPixelBuffer
    private func resizePixelBuffer(_ pixelBuffer: CVPixelBuffer, targetWidth: Int, targetHeight: Int) -> CVPixelBuffer? {
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!
        ] as CFDictionary

        var newPixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            targetWidth,
            targetHeight,
            kCVPixelFormatType_32ARGB,
            attrs,
            &newPixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = newPixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, [])
        CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags.readOnly)
        
        // Create contexts for both buffers
        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: targetWidth,
            height: targetHeight,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else {
            CVPixelBufferUnlockBaseAddress(buffer, [])
            CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags.readOnly)
            return nil
        }
        
        // Create CGImage from source pixel buffer
        let originalWidth = CVPixelBufferGetWidth(pixelBuffer)
        let originalHeight = CVPixelBufferGetHeight(pixelBuffer)
        
        guard let sourceContext = CGContext(
            data: CVPixelBufferGetBaseAddress(pixelBuffer),
            width: originalWidth,
            height: originalHeight,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ), let cgImage = sourceContext.makeImage() else {
            CVPixelBufferUnlockBaseAddress(buffer, [])
            CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags.readOnly)
            return nil
        }
        
        // Draw resized image
        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))
        
        CVPixelBufferUnlockBaseAddress(buffer, [])
        CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags.readOnly)
        
        return buffer
    }
    
    // Helper function to apply softmax to logits
    private func applySoftmax(logits: UnsafePointer<Float>, count: Int) -> [Float] {
        // Find the maximum value for numerical stability
        var maxValue: Float = -Float.infinity
        for i in 0..<count {
            maxValue = max(maxValue, logits[i])
        }
        
        // Calculate exp(logit - max) and sum
        var expValues: [Float] = []
        var sumExp: Float = 0.0
        for i in 0..<count {
            let expValue = expf(logits[i] - maxValue)
            expValues.append(expValue)
            sumExp += expValue
        }
        
        // Normalize to get probabilities
        var probabilities: [Float] = []
        for expValue in expValues {
            probabilities.append(expValue / sumExp)
        }
        
        return probabilities
    }

}