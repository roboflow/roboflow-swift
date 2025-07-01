//
//  LoadModel.swift
//  
//
//  Created by Maxwell Stone on 8/10/23.
//

import XCTest
import Roboflow
import CoreVideo
import CoreGraphics
import ImageIO
import Foundation

// cash counter api_key (already public)
var API_KEY = "fEto4us79wdzRJ2jkO6U"

final class LoadModel: XCTestCase {
    var model: RFModel?
    var classificationModel: RFClassificationModel?

    override func setUpWithError() throws {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }
    
    // Helper function to get ResNet model path from assets directory
    private func getResNetModelPath() -> URL? {
        let modelPath = "Tests/assets/resnet.mlpackage"
        let modelURL = URL(fileURLWithPath: modelPath)
        
        // Check if model file exists
        if FileManager.default.fileExists(atPath: modelURL.path) {
            return modelURL
        } else {
            XCTFail("ResNet model not found at \(modelPath). Please add ResNet.mlmodelc to the Tests/assets/ directory.")
            return nil
        }
    }

    // Helper function to load image and convert to CVPixelBuffer
    private func loadImageAsPixelBuffer(from imagePath: String) -> CVPixelBuffer? {
        let imageURL = URL(fileURLWithPath: imagePath)
        
        guard let imageSource = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) else {
            XCTFail("Failed to load test image from \(imagePath)")
            return nil
        }
        
        // Create CVPixelBuffer from CGImage
        let width = cgImage.width
        let height = cgImage.height
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!
        ] as CFDictionary

        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32ARGB,
            attrs,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            XCTFail("Failed to create pixel buffer")
            return nil
        }
        
        // Draw the CGImage into the pixel buffer
        CVPixelBufferLockBaseAddress(buffer, [])
        
        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else {
            CVPixelBufferUnlockBaseAddress(buffer, [])
            XCTFail("Failed to create graphics context")
            return nil
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        CVPixelBufferUnlockBaseAddress(buffer, [])
        
        return buffer
    }

    func testLoadModel() async {
        let rf = RoboflowMobile(apiKey: API_KEY)
        let (model, error, _, _) = await rf.load(model: "playing-cards-ow27d", modelVersion: 2)
        self.model = model
        XCTAssertNil(error)
        XCTAssertNotNil(model)
    }

    // test running inference
    func testRunInference() async {
        let rf = RoboflowMobile(apiKey: API_KEY)
        let (model, error, _, _) = await rf.load(model: "hard-hat-sample-txcpu", modelVersion: 6)
        self.model = model
        XCTAssertNil(error)
        XCTAssertNotNil(model)

        guard let buffer = loadImageAsPixelBuffer(from: "Tests/assets/hard-hat.jpeg"),
              let unwrappedModel = model else {
            XCTFail("Failed to load image or model is nil")
            return
        }
        
        let (results, inferenceError) = await unwrappedModel.detect(pixelBuffer: buffer)
        XCTAssertNil(inferenceError)
        // Note: predictions might be nil if no objects are detected in the test image, which is expected
        XCTAssertNotNil(results)
        XCTAssert(results?.count ?? 0 > 0)
    }
    
    func testLoadSeg() async {
        let rf = RoboflowMobile(apiKey: API_KEY)
        let (model, error, _, _) = await rf.load(model: "hat-1wxze-g6xvw", modelVersion: 1)
        self.model = model
        XCTAssertNil(error)
        XCTAssertNotNil(model)
    }
    
    // test running inference on segmentation model
    func testRunSegmentationInference() async {
        let rf = RoboflowMobile(apiKey: API_KEY)
        let (model, error, _, _) = await rf.load(model: "hat-1wxze-g6xvw", modelVersion: 1)
        self.model = model
        XCTAssertNil(error)
        XCTAssertNotNil(model)

        guard let buffer = loadImageAsPixelBuffer(from: "Tests/assets/cap.jpg"),
              let unwrappedModel = model else {
            XCTFail("Failed to load image or model is nil")
            return
        }
        
        let (results, inferenceError) = await unwrappedModel.detect(pixelBuffer: buffer)
        XCTAssertNil(inferenceError)
        // Note: predictions might be nil if no objects are detected in the test image, which is expected
        XCTAssertNotNil(results)
        XCTAssert(results?.count ?? 0 > 0)
    }
    
    // MARK: - ResNet Classification Tests
    
    func testLoadResNetModel() async {
        guard let modelURL = getResNetModelPath() else {
            XCTFail("Failed to get ResNet model path")
            return
        }
        
        let classificationModel = RFClassificationModel()
        let error = classificationModel.loadLocalModel(modelPath: modelURL)
        
        XCTAssertNil(error, "Failed to load ResNet model: \(error?.localizedDescription ?? "unknown error")")
        XCTAssertNotNil(classificationModel)
        
        // Configure the model
        classificationModel.configure(threshold: 0.1, overlap: 0.0, maxObjects: 0)
        
        self.classificationModel = classificationModel
    }
    
    func testResNetClassificationInference() async {
        guard let modelURL = getResNetModelPath() else {
            XCTFail("Failed to get ResNet model path")
            return
        }
        
        let classificationModel = RFClassificationModel()
        let loadError = classificationModel.loadLocalModel(modelPath: modelURL)
        
        XCTAssertNil(loadError, "Failed to load ResNet model")
        
        // Configure the model with low threshold to get more predictions
        classificationModel.configure(threshold: 0.01, overlap: 0.0, maxObjects: 0)
        
        // Use existing test image
        guard let buffer = loadImageAsPixelBuffer(from: "Tests/assets/hard-hat.jpeg") else {
            XCTFail("Failed to load test image")
            return
        }
        
        // Test detect method with CVPixelBuffer
        let (basePredictions, inferenceError) = await classificationModel.detect(pixelBuffer: buffer)
        
        XCTAssertNil(inferenceError, "Classification inference failed: \(inferenceError?.localizedDescription ?? "unknown error")")
        XCTAssertNotNil(basePredictions, "Predictions should not be nil")
        
        if let basePredictions = basePredictions {
            XCTAssertGreaterThan(basePredictions.count, 0, "Should have at least one prediction")
            
            // Cast to RFClassificationPrediction to test specific properties
            for basePrediction in basePredictions {
                guard let prediction = basePrediction as? RFClassificationPrediction else {
                    XCTFail("Prediction should be of type RFClassificationPrediction")
                    continue
                }
                
                XCTAssertFalse(prediction.className.isEmpty, "Class name should not be empty")
                XCTAssertGreaterThanOrEqual(prediction.confidence, 0.0, "Confidence should be >= 0")
                XCTAssertLessThanOrEqual(prediction.confidence, 1.0, "Confidence should be <= 1")
                XCTAssertGreaterThanOrEqual(prediction.classIndex, 0, "Class index should be >= 0")
                
                // Test getValues() method
                let values = prediction.getValues()
                XCTAssertNotNil(values["class"])
                XCTAssertNotNil(values["confidence"])
                XCTAssertNotNil(values["classIndex"])
            }
            
            // Verify we got meaningful results
            XCTAssertGreaterThan(basePredictions.count, 0, "Should have predictions")
            if let topBasePrediction = basePredictions.first,
               let topPrediction = topBasePrediction as? RFClassificationPrediction {
                XCTAssertGreaterThan(topPrediction.confidence, 0.1, "Top prediction should have reasonable confidence")
            }
        }
    }
    
    func testResNetDetectMethod() async {
        guard let modelURL = getResNetModelPath() else {
            XCTFail("Failed to get ResNet model path")
            return
        }
        
        let classificationModel = RFClassificationModel()
        let loadError = classificationModel.loadLocalModel(modelPath: modelURL)
        
        XCTAssertNil(loadError, "Failed to load ResNet model")
        
        // Configure the model
        classificationModel.configure(threshold: 0.01, overlap: 0.0, maxObjects: 0)
        
        // Use existing test image
        guard let buffer = loadImageAsPixelBuffer(from: "Tests/assets/cap.jpg") else {
            XCTFail("Failed to load test image")
            return
        }
        
        // Test detect method that returns RFPrediction objects (but are actually RFClassificationPrediction)
        let (basePredictions, inferenceError) = await classificationModel.detect(pixelBuffer: buffer)
        
        XCTAssertNil(inferenceError, "Detection inference failed: \(inferenceError?.localizedDescription ?? "unknown error")")
        XCTAssertNotNil(basePredictions, "Predictions should not be nil")
        
        if let basePredictions = basePredictions {
            XCTAssertGreaterThan(basePredictions.count, 0, "Should have at least one prediction")
            
            // Verify these can be cast to RFClassificationPrediction objects
            for basePrediction in basePredictions {
                guard let prediction = basePrediction as? RFClassificationPrediction else {
                    XCTFail("Prediction should be of type RFClassificationPrediction")
                    continue
                }
                
                XCTAssertFalse(prediction.className.isEmpty, "Class name should not be empty")
                XCTAssertGreaterThanOrEqual(prediction.confidence, 0.0, "Confidence should be >= 0")
                XCTAssertLessThanOrEqual(prediction.confidence, 1.0, "Confidence should be <= 1")
            }
            
            // Verify meaningful results  
            if let topBasePrediction = basePredictions.first,
               let topPrediction = topBasePrediction as? RFClassificationPrediction {
                XCTAssertGreaterThan(topPrediction.confidence, 0.1, "Top prediction should have reasonable confidence")
            }
        }
    }
    
    func testResNetGenericDetectMethod() async {
        guard let modelURL = getResNetModelPath() else {
            XCTFail("Failed to get ResNet model path")
            return
        }
        
        let classificationModel = RFClassificationModel()
        let loadError = classificationModel.loadLocalModel(modelPath: modelURL)
        
        XCTAssertNil(loadError, "Failed to load ResNet model")
        
        // Configure the model
        classificationModel.configure(threshold: 0.01, overlap: 0.0, maxObjects: 0)
        
        // Use existing test image
        guard let buffer = loadImageAsPixelBuffer(from: "Tests/assets/hard-hat.jpeg") else {
            XCTFail("Failed to load test image")
            return
        }
        
        // Test generic detect method that returns RFPrediction objects
        let (predictions, inferenceError) = await (classificationModel as RFModel).detect(pixelBuffer: buffer)
        
        XCTAssertNil(inferenceError, "Generic detection inference failed: \(inferenceError?.localizedDescription ?? "unknown error")")
        XCTAssertNotNil(predictions, "Predictions should not be nil")
        
        if let predictions = predictions {
            XCTAssertGreaterThan(predictions.count, 0, "Should have at least one prediction")
            
            // Verify these can be cast to RFClassificationPrediction
            for prediction in predictions {
                if let classificationPrediction = prediction as? RFClassificationPrediction {
                    XCTAssertFalse(classificationPrediction.className.isEmpty, "Class name should not be empty")
                    XCTAssertGreaterThanOrEqual(classificationPrediction.confidence, 0.0, "Confidence should be >= 0")
                    XCTAssertLessThanOrEqual(classificationPrediction.confidence, 1.0, "Confidence should be <= 1")
                } else {
                    XCTFail("Prediction should be castable to RFClassificationPrediction")
                }
            }
            
            // Verify meaningful results
            if let firstPrediction = predictions.first,
               let classificationPrediction = firstPrediction as? RFClassificationPrediction {
                XCTAssertGreaterThan(classificationPrediction.confidence, 0.1, "Top prediction should have reasonable confidence")
            }
        }
    }
    
    #if canImport(UIKit)
    func testResNetUIImageClassification() async {
        guard let modelURL = getResNetModelPath() else {
            XCTFail("Failed to get ResNet model path")
            return
        }
        
        let classificationModel = RFClassificationModel()
        let loadError = classificationModel.loadLocalModel(modelPath: modelURL)
        
        XCTAssertNil(loadError, "Failed to load ResNet model")
        
        // Configure the model
        classificationModel.configure(threshold: 0.01, overlap: 0.0, maxObjects: 0)
        
        // Load UIImage from test assets
        guard let image = loadUIImage(from: "Tests/assets/hard-hat.jpeg") else {
            XCTFail("Failed to load test image as UIImage")
            return
        }
        
        // Test detect method with UIImage
        let (basePredictions, inferenceError) = await classificationModel.detect(image: image)
        
        XCTAssertNil(inferenceError, "UIImage classification inference failed: \(inferenceError?.localizedDescription ?? "unknown error")")
        XCTAssertNotNil(basePredictions, "Predictions should not be nil")
        
        if let basePredictions = basePredictions {
            XCTAssertGreaterThan(basePredictions.count, 0, "Should have at least one prediction")
            
            // Test RFClassificationPrediction properties by casting
            for basePrediction in basePredictions {
                guard let prediction = basePrediction as? RFClassificationPrediction else {
                    XCTFail("Prediction should be of type RFClassificationPrediction")
                    continue
                }
                
                XCTAssertFalse(prediction.className.isEmpty, "Class name should not be empty")
                XCTAssertGreaterThanOrEqual(prediction.confidence, 0.0, "Confidence should be >= 0")
                XCTAssertLessThanOrEqual(prediction.confidence, 1.0, "Confidence should be <= 1")
                XCTAssertGreaterThanOrEqual(prediction.classIndex, 0, "Class index should be >= 0")
            }
            
            // Verify meaningful results
            if let topBasePrediction = basePredictions.first,
               let topPrediction = topBasePrediction as? RFClassificationPrediction {
                XCTAssertGreaterThan(topPrediction.confidence, 0.1, "Top prediction should have reasonable confidence")
            }
        }
    }
    #endif
}
