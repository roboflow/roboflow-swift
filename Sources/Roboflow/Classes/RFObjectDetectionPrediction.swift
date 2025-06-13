//
//  RFObjectDetectionPrediction.swift
//  Roboflow
//
//  Created by Nicholas Arner on 6/2/23.
//

import Foundation
import CoreGraphics

public class RFObjectDetectionPrediction: RFPrediction {
    let x: Float
    let y: Float
    let width: Float
    let height: Float
    let className: String
    let confidence: Float
    let color: CGColor
    let box: CGRect
    
    public init(x: Float, y: Float, width: Float, height: Float, className: String, confidence: Float, color: CGColor, box: CGRect) {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.className = className
        self.confidence = confidence
        self.color = color
        self.box = box
    }
    
    public override func getValues() -> [String: Any] {
        let rgbColor = [Int((color.components?[0])! * 255), Int((color.components?[1])! * 255), Int((color.components?[2])! * 255)]
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
