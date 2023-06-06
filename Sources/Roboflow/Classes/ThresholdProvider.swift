//
//  ThresholdProvider.swift
//  Roboflow
//
//  Created by Nicholas Arner on 6/2/23.
//

import Foundation
import CoreML

class ThresholdProvider: MLFeatureProvider {
    open var values = [
        "iouThreshold": MLFeatureValue(double: 0.5),
        "confidenceThreshold": MLFeatureValue(double: 0.4)
    ]
    var featureNames: Set<String> {
        return Set(values.keys)
    }
    func featureValue(for featureName: String) -> MLFeatureValue? {
        return values[featureName]
    }
}
