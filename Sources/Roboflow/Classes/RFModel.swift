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
@available(macOS 10.15, *)
public class RFModel: NSObject {

    public override init() {
        super.init()
    }
    
    //Stores the retreived ML model
    var mlModel: MLModel!
    var visionModel: VNCoreMLModel!
    var coreMLRequest: VNCoreMLRequest!
    
    //Configure the parameters for the model
    public func configure(threshold: Double, overlap: Double, maxObjects: Float, processingMode: ProcessingMode = .balanced, maxNumberPoints: Int = 500) {}
    
    //Load the retrieved CoreML model into an already created RFObjectDetectionModel instance
    func loadMLModel(modelPath: URL, colors: [String: String], classes: [String]) -> Error? {
        return nil
    }
    
    public func detect(pixelBuffer: CVPixelBuffer, completion: @escaping (([RFObjectDetectionPrediction]?, Error?) -> Void)) {
    }
 
    public func detect(pixelBuffer: CVPixelBuffer) async -> ([RFPrediction]?, Error?) {
        return await withCheckedContinuation { continuation in
            detect(pixelBuffer: pixelBuffer) { result, error in
                continuation.resume(returning: (result, error))
            }
        }
    }
}


