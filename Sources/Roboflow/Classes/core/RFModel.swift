//
//  RFObjectDetectionModel.swift
//  Roboflow
//
//  Created by Nicholas Arner on 4/12/22.
//

import Foundation
import CoreML
import Vision

public enum ProcessingMode {
    case quality // use full image resolution
    case balanced // use model resolution
    case performance // use half model resolution
}

//Creates an instance of an ML model that's hosted on Roboflow
public class RFModel: NSObject {

    public override init() {
        super.init()
    }
    
    //Stores the retreived ML model
    var mlModel: MLModel!
    var visionModel: VNCoreMLModel!
    var coreMLRequest: VNCoreMLRequest!
    var environment: [String: Any]!
    var modelPath: URL!
    var colors: [String: String]!
    var classes: [String]!
    var threshold: Double = 0.5
    var overlap: Double = 0.4
    var maxObjects: Float = 20
    var maskProcessingMode: ProcessingMode = .balanced
    var maskMaxNumberPoints: Int = 500

    //Configure the parameters for the model
    public func configure(threshold: Double = 0.5, overlap: Double = 0.5, maxObjects: Float = 20, processingMode: ProcessingMode = .balanced, maxNumberPoints: Int = 500) {
        self.threshold = threshold
        self.overlap = overlap
        self.maxObjects = maxObjects
        self.maskProcessingMode = processingMode
        self.maskMaxNumberPoints = maxNumberPoints
    }
    
    //Load the retrieved CoreML model into an already created RFObjectDetectionModel instance
    func loadMLModel(modelPath: URL, colors: [String: String], classes: [String], environment: [String: Any]) -> Error? {
        self.environment = environment
        self.modelPath = modelPath
        self.colors = colors
        self.classes = classes
        return nil
    }
    
    public func detect(pixelBuffer: CVPixelBuffer, completion: @escaping (([RFPrediction]?, Error?) -> Void)) {
        completion(nil, NSError(domain: "RFModel", code: -1, userInfo: [NSLocalizedDescriptionKey: "Error loading model"]))
    }
 
    public func detect(pixelBuffer: CVPixelBuffer) async -> ([RFPrediction]?, Error?) {
        if #available(macOS 10.15, *) {
            return await withCheckedContinuation { continuation in
                detect(pixelBuffer: pixelBuffer) { result, error in
                    continuation.resume(returning: (result, error))
                }
            }
        } else {
            // Fallback on earlier versions
            return (nil, UnsupportedOSError())
        }
    }
}


