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
#if canImport(UIKit)
import UIKit
#endif

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
        let modelPath = "Tests/assets/ResNet.mlmodelc"
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
    
    #if canImport(UIKit)
    // Helper function to load UIImage for UIKit tests
    private func loadUIImage(from imagePath: String) -> UIImage? {
        return UIImage(contentsOfFile: imagePath)
    }
    #endif

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
        
        // Test classify method with CVPixelBuffer
        let (predictions, inferenceError) = await classificationModel.classify(pixelBuffer: buffer)
        
        XCTAssertNil(inferenceError, "Classification inference failed: \(inferenceError?.localizedDescription ?? "unknown error")")
        XCTAssertNotNil(predictions, "Predictions should not be nil")
        
        if let predictions = predictions {
            XCTAssertGreaterThan(predictions.count, 0, "Should have at least one prediction")
            
            // Test RFClassificationPrediction properties
            for prediction in predictions {
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
            
            print("ResNet Classification Results:")
            for (index, prediction) in predictions.prefix(5).enumerated() {
                print("  \(index + 1). \(prediction.className) - \(String(format: "%.3f", prediction.confidence))")
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
        
        // Test detect method that returns RFClassificationPrediction objects
        let (predictions, inferenceError) = await classificationModel.detect(pixelBuffer: buffer)
        
        XCTAssertNil(inferenceError, "Detection inference failed: \(inferenceError?.localizedDescription ?? "unknown error")")
        XCTAssertNotNil(predictions, "Predictions should not be nil")
        
        if let predictions = predictions {
            XCTAssertGreaterThan(predictions.count, 0, "Should have at least one prediction")
            
            // Verify these are RFClassificationPrediction objects
            for prediction in predictions {
                XCTAssertTrue(prediction is RFClassificationPrediction, "Should be RFClassificationPrediction object")
                XCTAssertFalse(prediction.className.isEmpty, "Class name should not be empty")
                XCTAssertGreaterThanOrEqual(prediction.confidence, 0.0, "Confidence should be >= 0")
                XCTAssertLessThanOrEqual(prediction.confidence, 1.0, "Confidence should be <= 1")
            }
            
            print("ResNet Detection Results:")
            for (index, prediction) in predictions.prefix(3).enumerated() {
                print("  \(index + 1). \(prediction.className) - \(String(format: "%.3f", prediction.confidence))")
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
            
            print("ResNet Generic Detection Results:")
            for (index, prediction) in predictions.prefix(3).enumerated() {
                if let classificationPrediction = prediction as? RFClassificationPrediction {
                    print("  \(index + 1). \(classificationPrediction.className) - \(String(format: "%.3f", classificationPrediction.confidence))")
                }
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
        
        // Test classify method with UIImage
        let (predictions, inferenceError) = await classificationModel.classify(image: image)
        
        XCTAssertNil(inferenceError, "UIImage classification inference failed: \(inferenceError?.localizedDescription ?? "unknown error")")
        XCTAssertNotNil(predictions, "Predictions should not be nil")
        
        if let predictions = predictions {
            XCTAssertGreaterThan(predictions.count, 0, "Should have at least one prediction")
            
            // Test RFClassificationPrediction properties
            for prediction in predictions {
                XCTAssertFalse(prediction.className.isEmpty, "Class name should not be empty")
                XCTAssertGreaterThanOrEqual(prediction.confidence, 0.0, "Confidence should be >= 0")
                XCTAssertLessThanOrEqual(prediction.confidence, 1.0, "Confidence should be <= 1")
                XCTAssertGreaterThanOrEqual(prediction.classIndex, 0, "Class index should be >= 0")
            }
            
            print("ResNet UIImage Classification Results:")
            for (index, prediction) in predictions.prefix(3).enumerated() {
                print("  \(index + 1). \(prediction.className) - \(String(format: "%.3f", prediction.confidence))")
            }
        }
        
        // Test detect method with UIImage
        let (detectPredictions, detectError) = await classificationModel.detect(image: image)
        
        XCTAssertNil(detectError, "UIImage detect inference failed")
        XCTAssertNotNil(detectPredictions, "Detect predictions should not be nil")
        
        if let detectPredictions = detectPredictions {
            XCTAssertGreaterThan(detectPredictions.count, 0, "Should have at least one detect prediction")
            
            // Verify results are consistent between classify and detect methods
            XCTAssertEqual(predictions?.count, detectPredictions.count, "Classify and detect should return same number of predictions")
        }
    }
    #endif
}
