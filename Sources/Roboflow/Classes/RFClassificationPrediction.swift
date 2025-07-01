//
//  RFClassificationPrediction.swift
//  Roboflow
//
//  Created by Maxwell Stone on 6/16/25.
//

import Foundation
import CoreGraphics

public class RFClassificationPrediction: RFPrediction {
    public let className: String
    public let confidence: Float
    public let classIndex: Int
    
    public init(className: String, confidence: Float, classIndex: Int) {
        self.className = className
        self.confidence = confidence
        self.classIndex = classIndex
    }
    
    public override func getValues() -> [String: Any] {
        let result = [
            "confidence": Double(confidence),
            "class": className,
            "classIndex": classIndex
        ] as [String: Any]
        
        return result
    }
}