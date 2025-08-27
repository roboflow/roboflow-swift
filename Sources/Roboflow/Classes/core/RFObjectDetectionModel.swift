//
//  RFObjectDetectionModel.swift
//  Roboflow
//
//  Created by Nicholas Arner on 4/12/22.
//

import Foundation
import CoreML
import Vision
//Creates an instance of an ML model that's hosted on Roboflow
public class RFObjectDetectionModel: RFModel {

    public override init() {
        super.init()
    }
    
    //Stores the retreived ML model
    var thresholdProvider = ThresholdProvider()
    
    //Configure the parameters for the model
    public override func configure(threshold: Double = 0.5, overlap: Double = 0.5, maxObjects: Float = 20, processingMode: ProcessingMode = .balanced, maxNumberPoints: Int = 500) {
        super.configure(threshold: threshold, overlap: overlap, maxObjects: maxObjects, processingMode: processingMode, maxNumberPoints: maxNumberPoints)
        thresholdProvider.values = ["iouThreshold": MLFeatureValue(double: self.overlap),
                                    "confidenceThreshold": MLFeatureValue(double: self.threshold)]
        if visionModel != nil {
            if #available(macOS 10.15, *) {
                visionModel.featureProvider = thresholdProvider
            } else {
                // Fallback on earlier versions
            }
        }
    }
    
    //Load the retrieved CoreML model into an already created RFObjectDetectionModel instance
    override func loadMLModel(modelPath: URL, colors: [String: String], classes: [String], environment: [String: Any]) -> Error? {
        let _ = super.loadMLModel(modelPath: modelPath, colors: colors, classes: classes, environment: environment)
        do {
            if #available(macOS 10.14, *) {
                let config = MLModelConfiguration()
                if #available(macOS 10.15, *) {
                    mlModel = try yolov5s(contentsOf: modelPath, configuration: config).model
                } else {
                    // Fallback on earlier versions
                    return UnsupportedOSError()
                }
                visionModel = try VNCoreMLModel(for: mlModel)
                if #available(macOS 10.15, *) {
                    visionModel.featureProvider = thresholdProvider
                } else {
                    // Fallback on earlier versions
                    return UnsupportedOSError()
                }
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
    
    //Run image through model and return Detections
    @available(*, renamed: "detect(image:)")
    public override func detect(pixelBuffer buffer: CVPixelBuffer, completion: @escaping (([RFPrediction]?, Error?) -> Void)) {
        guard let coreMLRequest = self.coreMLRequest else {
            completion(nil, "Model initialization failed.")
            return
        }
        let handler = VNImageRequestHandler(cvPixelBuffer: buffer)

        do {
            try handler.perform([coreMLRequest])
            
            guard let detectResults = coreMLRequest.results as? [VNDetectedObjectObservation] else { return }
            
            var detections:[RFObjectDetectionPrediction] = []
            for detectResult in detectResults {
                let flippedBox = CGRect(x: detectResult.boundingBox.minX, y: 1 - detectResult.boundingBox.maxY, width: detectResult.boundingBox.width, height: detectResult.boundingBox.height)
                
                let box = VNImageRectForNormalizedRect(flippedBox, Int(buffer.width()), Int(buffer.height()))
                let confidence = detectResult.confidence
                var label:String = ""
                if #available(macOS 10.14, *) {
                    if let recognizedResult = detectResult as? VNRecognizedObjectObservation, let classLabel = recognizedResult.labels.first?.identifier {
                        // class labels may be stripped from the model when weights trained externally and uploaded.
                        // if our class it is an integer and it is not a defined class, look it up in our class map.
                        if let intValue = Int(classLabel), !classes.contains(classLabel), intValue < classes.count {
                            label = classes[intValue]
                        } else {
                            label = classLabel
                        }
                    }
                } else {
                    // Fallback on earlier versions
                    completion(nil, UnsupportedOSError())
                    return
                }
                let detection = RFObjectDetectionPrediction(x: Float((box.maxX+box.minX)/2.0), y: Float((box.maxY+box.minY)/2.0), width: Float((box.maxX-box.minX)), height: Float((box.maxY-box.minY)), className: label, confidence: confidence, color: hexStringToCGColor(hex: colors[label] ?? "#ff0000"), box: box)
                detections.append(detection)
            }
            completion(detections, nil)
        } catch let error {
            completion(nil, error)
        }
    }
}
