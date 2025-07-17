//
//  RFObjectDetectionPrediction.swift
//  Roboflow
//
//  Created by Nicholas Arner on 6/2/23.
//

import Foundation
import CoreGraphics

public class RFObjectDetectionPrediction: RFPrediction {
    public let x: Float
    public let y: Float
    public let width: Float
    public let height: Float
    public let className: String
    public let confidence: Float
    public let color: CGColor
    public let box: CGRect
    
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
