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
    var threshold: Double = 0.5
    var classes: [String] = []
    
    //Configure the parameters for the model
    public override func configure(threshold: Double, overlap: Double, maxObjects: Float, processingMode: ProcessingMode = .balanced, maxNumberPoints: Int = 500) {
        self.threshold = threshold
    }
    
    //Load the retrieved CoreML model into an already created RFClassificationModel instance
    override func loadMLModel(modelPath: URL, colors: [String: String], classes: [String]) -> Error? {
        self.classes = classes
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
    
    //Load a local model file (for manually placed models like ResNet)
    public func loadLocalModel(modelPath: URL) -> Error? {
        do {
            if #available(macOS 10.14, *) {
                let config = MLModelConfiguration()
                if #available(macOS 10.15, *) {
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
                    
                    mlModel = try MLModel(contentsOf: modelURL, configuration: config)
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
            completion(nil, "Model initialization failed.")
            return
        }
        

        
        do {
            // Get the actual input name from model description
            guard let inputName = mlModel.modelDescription.inputDescriptionsByName.keys.first,
                  let inputDescription = mlModel.modelDescription.inputDescriptionsByName[inputName] else {
                completion(nil, "Could not find model input name")
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
                completion(nil, "Could not find model output")
                return
            }
            
            var predictions: [RFPrediction] = []
            
            // Handle different output types
            if output.type == .dictionary, let probDict = output.dictionaryValue as? [String: NSNumber] {
                // Dictionary output (class name -> probability)
                for (className, probValue) in probDict {
                    let confidence = probValue.floatValue
                    if confidence >= Float(threshold) {
                        let prediction = RFClassificationPrediction(
                            className: className,
                            confidence: confidence,
                            classIndex: 0 // Index not available in dictionary output
                        )
                        predictions.append(prediction)
                    }
                }
            } else if let multiArray = output.multiArrayValue {
                // Multi-array output (probabilities from model)
                let probabilities = multiArray.dataPointer.bindMemory(to: Float.self, capacity: multiArray.count)
                
                for i in 0..<multiArray.count {
                    let confidence = probabilities[i]
                    if confidence >= Float(threshold) {
                        let prediction = RFClassificationPrediction(
                            className: "class_\(i)", // Generic class name
                            confidence: confidence,
                            classIndex: i
                        )
                        predictions.append(prediction)
                    }
                }
            }
            
            // Sort by confidence (highest first) - cast to RFClassificationPrediction since that's what we create
            predictions.sort { 
                guard let pred1 = $0 as? RFClassificationPrediction, 
                      let pred2 = $1 as? RFClassificationPrediction else { 
                    return false 
                }
                return pred1.confidence > pred2.confidence
            }
            
            completion(predictions, nil)
            
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

}