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
    
    //Run image through model and return Classification predictions as Object Detection predictions for compatibility
    public override func detect(pixelBuffer buffer: CVPixelBuffer, completion: @escaping (([RFObjectDetectionPrediction]?, Error?) -> Void)) {
        classify(pixelBuffer: buffer) { [weak self] predictions, error in
            // Convert classification predictions to object detection predictions for compatibility with base class
            let objectDetectionPredictions = predictions?.map { prediction in
                // Create a full-image bounding box for classification results
                let box = CGRect(x: 0, y: 0, width: Int(buffer.width()), height: Int(buffer.height()))
                return RFObjectDetectionPrediction(
                    x: Float(buffer.width()) / 2.0,
                    y: Float(buffer.height()) / 2.0,
                    width: Float(buffer.width()),
                    height: Float(buffer.height()),
                    className: prediction.className,
                    confidence: prediction.confidence,
                    color: hexStringToCGColor(hex: "#00ff00"), // Default green color
                    box: box
                )
            }
            completion(objectDetectionPredictions, error)
        }
    }
    
    //Run image through model and return Classification predictions directly
    public func detect(pixelBuffer buffer: CVPixelBuffer, completion: @escaping (([RFClassificationPrediction]?, Error?) -> Void)) {
        classify(pixelBuffer: buffer, completion: completion)
    }
    
    //Async version that returns RFClassificationPrediction objects as RFPrediction
    public override func detect(pixelBuffer: CVPixelBuffer) async -> ([RFPrediction]?, Error?) {
        if #available(macOS 10.15, *) {
            return await withCheckedContinuation { continuation in
                classify(pixelBuffer: pixelBuffer) { predictions, error in
                    // Return RFClassificationPrediction objects as RFPrediction
                    let rfPredictions = predictions?.map { $0 as RFPrediction }
                    continuation.resume(returning: (rfPredictions, error))
                }
            }
        } else {
            // Fallback on earlier versions
            return (nil, UnsupportedOSError())
        }
    }
    
    //Run image through model and return Classification predictions
    public func classify(pixelBuffer buffer: CVPixelBuffer, completion: @escaping (([RFClassificationPrediction]?, Error?) -> Void)) {
        guard let coreMLRequest = self.coreMLRequest else {
            completion(nil, "Model initialization failed.")
            return
        }
        let handler = VNImageRequestHandler(cvPixelBuffer: buffer)

        do {
            try handler.perform([coreMLRequest])
            
            guard let classificationResults = coreMLRequest.results as? [VNClassificationObservation] else { 
                completion(nil, "Unable to get classification results from model")
                return 
            }
            
            var predictions: [RFClassificationPrediction] = []
            for (index, result) in classificationResults.enumerated() {
                if result.confidence >= Float(threshold) {
                    let prediction = RFClassificationPrediction(
                        className: result.identifier,
                        confidence: result.confidence,
                        classIndex: index
                    )
                    predictions.append(prediction)
                }
            }
            
            // Sort by confidence (highest first)
            predictions.sort { $0.confidence > $1.confidence }
            
            completion(predictions, nil)
        } catch let error {
            completion(nil, error)
        }
    }
    
    //Async version that returns RFClassificationPrediction objects
    public func classify(pixelBuffer buffer: CVPixelBuffer) async -> ([RFClassificationPrediction]?, Error?) {
        if #available(macOS 10.15, *) {
            return await withCheckedContinuation { continuation in
                classify(pixelBuffer: buffer) { predictions, error in
                    continuation.resume(returning: (predictions, error))
                }
            }
        } else {
            // Fallback on earlier versions
            return (nil, UnsupportedOSError())
        }
    }
}