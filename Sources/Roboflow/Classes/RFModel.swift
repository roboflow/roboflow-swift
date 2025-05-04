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
public class RFModel: NSObject {

    public override init() {
        super.init()
    }
    
    //Stores the retreived ML model
    var mlModel: MLModel!
    var visionModel: VNCoreMLModel!
    var coreMLRequest: VNCoreMLRequest!
    
    //Configure the parameters for the model
    public func configure(threshold: Double, overlap: Double, maxObjects: Float) {}
    
    //Load the retrieved CoreML model into an already created RFObjectDetectionModel instance
    func loadMLModel(modelPath: URL, colors: [String: String], classes: [String]) -> Error? {
        return nil
    }
    
    //Run image through model and return Detections
    @available(*, renamed: "detect(image:)")
    public func detect(image:UIImage, completion: @escaping (([RFObjectDetectionPrediction]?, Error?) -> Void)) {
    }
    
    public func detect(image: UIImage) async -> ([RFPrediction]?, Error?) {
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
 
    public func detect(pixelBuffer: CVPixelBuffer) async -> ([RFPrediction]?, Error?) {
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
