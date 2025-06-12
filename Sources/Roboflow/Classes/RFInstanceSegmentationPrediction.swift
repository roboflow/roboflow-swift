//
//  RFObjectDetectionPrediction.swift
//  Roboflow
//
//  Created by Nicholas Arner on 6/2/23.
//

import Foundation
import CoreGraphics

public class RFInstanceSegmentationPrediction: RFObjectDetectionPrediction {
    var points: [CGPoint]
    var mask: [[UInt8]]?
    
    public init(x: Float, y: Float, width: Float, height: Float, className: String, confidence: Float, color: CGColor, box: CGRect, points: [CGPoint], mask: [[UInt8]]?) {
        self.points = points
        self.mask = mask ?? []
        super.init(x: x, y: y, width: width, height: height, className: className, confidence: confidence, color: color, box: box)
    }
    
    public override func getValues() -> [String: Any] {
        var result = super.getValues()
        let poly: [[String:Float]] = points.map { x in
            return ["x": Float(x.x), "y": Float(x.y)]
        }
        result["points"] = poly
        if let mask = mask {
            result["mask"] = mask
        }
        
        return result
    }

}
