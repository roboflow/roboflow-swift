//
//  RFObjectDetectionPrediction.swift
//  Roboflow
//
//  Created by Nicholas Arner on 6/2/23.
//

import Foundation
import UIKit

public struct RFObjectDetectionPrediction {
    let x: Float
    let y: Float
    let width: Float
    let height: Float
    let className: String
    let confidence: Float
    let color: UIColor
    let box: CGRect
    
    public func getValues() -> [String: Any] {
        let ciColor = CIColor(color: color)
        let rgbColor = [Int(ciColor.red * 255), Int(ciColor.green * 255), Int(ciColor.blue * 255)]
        let result = [
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "confidence": Double(confidence),
            "class": className,
            "color": rgbColor
        ] as [String: Any]
        
        return result
    }

}
