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
        
        // Perform classification using async/await
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
            
            // Print results
            print("Classification Results:")
            for prediction in predictions {
                print("Class: \(prediction.className), Confidence: \(prediction.confidence), Index: \(prediction.classIndex)")
            }
            
            // Get the top prediction
            if let topPrediction = predictions.first {
                print("Top prediction: \(topPrediction.className) with confidence \(topPrediction.confidence)")
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
        
        // Perform classification using completion handler
        classificationModel.classify(image: image) { predictions, error in
            if let error = error {
                print("Classification error: \(error)")
                return
            }
            
            guard let predictions = predictions else {
                print("No predictions returned")
                return
            }
            
            print("Classification Results:")
            for prediction in predictions {
                print("Class: \(prediction.className), Confidence: \(prediction.confidence)")
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
                // Handle results...
                if let predictions = predictions {
                    for prediction in predictions {
                        print("Class: \(prediction.className), Confidence: \(prediction.confidence)")
                    }
                }
            }
        }
    }
}