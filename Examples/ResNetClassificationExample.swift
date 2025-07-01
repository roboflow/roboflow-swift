//
//  ResNetClassificationExample.swift
//  Roboflow
//
//  Created by Maxwell Stone on 6/16/25.
//

import Foundation
import UIKit
import Roboflow

class ResNetClassificationExample {
    
    func loadAndUseResNetModel() {
        // Initialize Roboflow with your API key
        let rf = RoboflowMobile(apiKey: "your_api_key_here")
        
        // Create a classification model instance
        let classificationModel = RFClassificationModel()
        
        // Load your ResNet model from the provided mlmodelc package
        // Make sure to add the .mlmodelc package to your app bundle
        guard let modelURL = Bundle.main.url(forResource: "ResNet", withExtension: "mlmodelc") else {
            print("Could not find ResNet.mlmodelc in app bundle")
            return
        }
        
        // Load the local model
        if let error = classificationModel.loadLocalModel(modelPath: modelURL) {
            print("Error loading model: \(error)")
            return
        }
        
        // Configure the model (set confidence threshold)
        classificationModel.configure(threshold: 0.3, overlap: 0.0, maxObjects: 0)
        
        // Load an image to classify
        guard let image = UIImage(named: "your_image.jpg") else {
            print("Could not load image")
            return
        }
        
        // Option 1: Use classify() method - returns RFClassificationPrediction objects
        Task {
            let (predictions, error) = await classificationModel.classify(image: image)
            
            if let error = error {
                print("Classification error: \(error)")
                return
            }
            
            guard let predictions = predictions else {
                print("No predictions returned")
                return
            }
            
            // Print results using RFClassificationPrediction properties
            print("Classification Results:")
            for prediction in predictions {
                print("Class: \(prediction.className)")
                print("Confidence: \(String(format: "%.3f", prediction.confidence))")
                print("Class Index: \(prediction.classIndex)")
                print("Raw Values: \(prediction.getValues())")
                print("---")
            }
            
            // Get the top prediction (highest confidence)
            if let topPrediction = predictions.first {
                print("🏆 Top prediction: \(topPrediction.className) with confidence \(String(format: "%.3f", topPrediction.confidence))")
            }
            
                         // Filter predictions above a certain threshold
             let highConfidencePredictions = predictions.filter { $0.confidence > 0.7 }
             print("High confidence predictions (>70%): \(highConfidencePredictions.count)")
        }
        
        // Option 2: Use detect() method - also returns RFClassificationPrediction objects
        Task {
            let (predictions, error) = await classificationModel.detect(image: image)
            
            if let error = error {
                print("Detection error: \(error)")
                return
            }
            
            guard let predictions = predictions else {
                print("No predictions returned")
                return
            }
            
            print("Detection method results (RFClassificationPrediction objects):")
            for prediction in predictions {
                print("Class: \(prediction.className)")
                print("Confidence: \(String(format: "%.3f", prediction.confidence))")
                print("Class Index: \(prediction.classIndex)")
            }
        }
    }
    
    func loadAndUseResNetModelWithCallback() {
        // Alternative approach using completion handler
        let classificationModel = RFClassificationModel()
        
        guard let modelURL = Bundle.main.url(forResource: "ResNet", withExtension: "mlmodelc") else {
            print("Could not find ResNet.mlmodelc in app bundle")
            return
        }
        
        if let error = classificationModel.loadLocalModel(modelPath: modelURL) {
            print("Error loading model: \(error)")
            return
        }
        
        classificationModel.configure(threshold: 0.3, overlap: 0.0, maxObjects: 0)
        
        guard let image = UIImage(named: "your_image.jpg") else {
            print("Could not load image")
            return
        }
        
        // Perform classification using completion handler - returns RFClassificationPrediction objects
        classificationModel.classify(image: image) { predictions, error in
            if let error = error {
                print("Classification error: \(error)")
                return
            }
            
            guard let predictions = predictions else {
                print("No predictions returned")
                return
            }
            
            print("Classification Results using RFClassificationPrediction:")
            for prediction in predictions {
                print("Class: \(prediction.className)")
                print("Confidence: \(String(format: "%.3f", prediction.confidence))")
                print("Class Index: \(prediction.classIndex)")
            }
        }
    }
    
    func loadResNetFromRoboflowAPI() {
        // If your ResNet model is hosted on Roboflow, you can load it like this:
        let rf = RoboflowMobile(apiKey: "your_api_key_here")
        
        rf.load(model: "your-resnet-model", modelVersion: 1) { model, error, modelName, modelType in
            if let error = error {
                print("Error loading model from API: \(error)")
                return
            }
            
            guard let classificationModel = model as? RFClassificationModel else {
                print("Loaded model is not a classification model")
                return
            }
            
            // Configure the model
            classificationModel.configure(threshold: 0.3, overlap: 0.0, maxObjects: 0)
            
            // Use the model for classification...
            guard let image = UIImage(named: "your_image.jpg") else {
                print("Could not load image")
                return
            }
            
            classificationModel.classify(image: image) { predictions, error in
                // Handle results using RFClassificationPrediction objects
                if let predictions = predictions {
                    for prediction in predictions {
                        print("Class: \(prediction.className)")
                        print("Confidence: \(String(format: "%.3f", prediction.confidence))")
                        print("Class Index: \(prediction.classIndex)")
                    }
                }
            }
        }
    }
    
    func useGenericDetectMethod() {
        // Example showing how to use the generic detect method that returns RFPrediction objects
        let classificationModel = RFClassificationModel()
        
        guard let modelURL = Bundle.main.url(forResource: "ResNet", withExtension: "mlmodelc") else {
            print("Could not find ResNet.mlmodelc in app bundle")
            return
        }
        
        if let error = classificationModel.loadLocalModel(modelPath: modelURL) {
            print("Error loading model: \(error)")
            return
        }
        
        classificationModel.configure(threshold: 0.3, overlap: 0.0, maxObjects: 0)
        
        guard let image = UIImage(named: "your_image.jpg") else {
            print("Could not load image")
            return
        }
        
        // Convert UIImage to CVPixelBuffer for the generic detect method
        let size = image.size
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!
        ] as CFDictionary

        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(size.width),
            Int(size.height),
            kCVPixelFormatType_32ARGB,
            attrs,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            print("Failed to create pixel buffer")
            return
        }
        
        // Use the generic detect method that returns RFPrediction objects
        Task {
            let (predictions, error) = await classificationModel.detect(pixelBuffer: buffer)
            
            if let error = error {
                print("Detection error: \(error)")
                return
            }
            
            guard let predictions = predictions else {
                print("No predictions returned")
                return
            }
            
            print("Generic detect method results:")
            for prediction in predictions {
                // Cast RFPrediction back to RFClassificationPrediction to access specific properties
                if let classificationPrediction = prediction as? RFClassificationPrediction {
                    print("Class: \(classificationPrediction.className)")
                    print("Confidence: \(String(format: "%.3f", classificationPrediction.confidence))")
                    print("Class Index: \(classificationPrediction.classIndex)")
                    print("Values: \(classificationPrediction.getValues())")
                } else {
                    print("Generic prediction: \(prediction.getValues())")
                }
            }
        }
    }
}