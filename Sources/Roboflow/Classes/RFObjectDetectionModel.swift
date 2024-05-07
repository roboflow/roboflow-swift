//
//  RFObjectDetectionModel.swift
//  Roboflow
//
//  Created by Nicholas Arner on 4/12/22.
//

import Foundation
import CoreML
import Vision
import UIKit
//Creates an instance of an ML model that's hosted on Roboflow
public class RFObjectDetectionModel: NSObject {

    public override init() {
        super.init()
    }
    
    //Default model configuration parameters
    var threshold: Double = 0.5
    var overlap: Double = 0.5
    var maxObjects: Float = 20.0
    var colors: [String: String]!
    //Stores the retreived ML model
    var mlModel: MLModel!
    var visionModel: VNCoreMLModel!
    var coreMLRequest: VNCoreMLRequest!
    private var thresholdProvider = ThresholdProvider()
    
    //Configure the parameters for the model
    public func configure(threshold: Double, overlap: Double, maxObjects: Float) {
        self.threshold = threshold
        self.overlap = overlap
        self.maxObjects = maxObjects
        thresholdProvider.values = ["iouThreshold": MLFeatureValue(double: self.overlap),
                                    "confidenceThreshold": MLFeatureValue(double: self.threshold)]
        if visionModel != nil {
            visionModel.featureProvider = thresholdProvider
        }
    }
    
    //Load the retrieved CoreML model into an already created RFObjectDetectionModel instance
    func loadMLModel(modelPath: URL, colors: [String: String]) -> Error? {
        self.colors = colors
        do {
            let config = MLModelConfiguration()
            mlModel = try yolov5s(contentsOf: modelPath, configuration: config).model
            visionModel = try VNCoreMLModel(for: mlModel)
            visionModel.featureProvider = thresholdProvider
            let request = VNCoreMLRequest(model: visionModel)
            request.imageCropAndScaleOption = .scaleFill
            coreMLRequest = request
        } catch {
            return error
        }
        return nil
    }
    
    //Run image through model and return Detections
    @available(*, renamed: "detect(image:)")
    public func detect(image:UIImage, completion: @escaping (([RFObjectDetectionPrediction]?, Error?) -> Void)) {
        guard let coreMLRequest = self.coreMLRequest else {
            completion(nil, "Model initialization failed.")
            return
        }
        guard let ciImage = CIImage(image: image) else {
            completion(nil, "Image failed.")
            return
        }
        let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])

        do {
            try handler.perform([coreMLRequest])
            
            guard let detectResults = coreMLRequest.results as? [VNDetectedObjectObservation] else { return }
            
            var detections:[RFObjectDetectionPrediction] = []
            for detectResult in detectResults {
                let flippedBox = CGRect(x: detectResult.boundingBox.minX, y: 1 - detectResult.boundingBox.maxY, width: detectResult.boundingBox.width, height: detectResult.boundingBox.height)
                
                let box = VNImageRectForNormalizedRect(flippedBox, Int(image.size.width), Int(image.size.height))
                let confidence = detectResult.confidence
                var label:String = ""
                if let recognizedResult = detectResult as? VNRecognizedObjectObservation, let classLabel = recognizedResult.labels.first?.identifier {
                    label = classLabel
                }
                let detection = RFObjectDetectionPrediction(x: Float((box.maxX+box.minX)/2.0), y: Float((box.maxY+box.minY)/2.0), width: Float((box.maxX-box.minX)), height: Float((box.maxY-box.minY)), className: label, confidence: confidence, color: hexStringToUIColor(hex: colors[label] ?? "#ff0000"), box: box)
                detections.append(detection)
            }
            completion(detections, nil)
        } catch let error {
            completion(nil, error)
        }
    }
    
    public func detect(image: UIImage) async -> ([RFObjectDetectionPrediction]?, Error?) {
        return await withCheckedContinuation { continuation in
            detect(image: image) { result, error in
                continuation.resume(returning: (result, error))
            }
        }
    }
    
    public func detect(pixelBuffer: CVPixelBuffer, completion: @escaping (([RFObjectDetectionPrediction]?, Error?) -> Void)) {
        let image = UIImage(pixelBuffer: pixelBuffer)
        detect(image: image!) { detections, error in
            completion(detections, nil)
        }
    }
 
    public func detect(pixelBuffer: CVPixelBuffer) async -> ([RFObjectDetectionPrediction]?, Error?) {
        return await withCheckedContinuation { continuation in
            detect(pixelBuffer: pixelBuffer) { result, error in
                continuation.resume(returning: (result, error))
            }
        }
    }
}

func hexStringToUIColor (hex:String) -> UIColor {
    var cString:String = hex.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()

    if (cString.hasPrefix("#")) {
        cString.remove(at: cString.startIndex)
    }

    if ((cString.count) != 6) {
        return UIColor.gray
    }

    var rgbValue:UInt64 = 0
    Scanner(string: cString).scanHexInt64(&rgbValue)

    return UIColor(
        red: CGFloat((rgbValue & 0xFF0000) >> 16) / 255.0,
        green: CGFloat((rgbValue & 0x00FF00) >> 8) / 255.0,
        blue: CGFloat(rgbValue & 0x0000FF) / 255.0,
        alpha: CGFloat(1.0)
    )
}
