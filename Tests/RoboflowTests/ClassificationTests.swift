//
//  ClassificationTests.swift
//  
//
//  Created by Maxwell Stone on 12/19/24.
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

final class ClassificationTests: XCTestCase {
    var model: RFModel?

    override func setUpWithError() throws {
        // Put setup code here. This method is called before the invocation of each test method in the class.
        let rf = RoboflowMobile(apiKey: API_KEY)
        rf.clearModelCache(modelName: "banana-ripeness-frqdw", modelVersion: 6)
        rf.clearModelCache(modelName: "banana-ripeness-frqdw", modelVersion: 5)
    }

    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }
    
    // MARK: - Classification Model Tests
    
    func testLoadBananasModel() async {
        guard let model = await TestUtils.loadBananasModel() else {
            XCTFail("Failed to load banana ripeness model")
            return
        }
        
        // Configure the model
        model.configure(threshold: 0.1, overlap: 0.0, maxObjects: 0)
        
        self.model = model
    }
    
    func testBananasClassificationInference() async {
        guard let model = await TestUtils.loadBananasModel() else {
            XCTFail("Failed to load banana ripeness model")
            return
        }
        
        // Configure the model with low threshold to get more predictions
        model.configure(threshold: 0.01, overlap: 0.0, maxObjects: 0)
        
        // Use banana image for testing
        guard let buffer = TestUtils.loadImageAsPixelBuffer(from: "Tests/assets/banana.jpg") else {
            XCTFail("Failed to load banana test image")
            return
        }
        
        // Test detect method with CVPixelBuffer
        let (basePredictions, inferenceError) = await model.detect(pixelBuffer: buffer)
        
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
                XCTAssertGreaterThanOrEqual(prediction.classId, 0, "Class index should be >= 0")
                
                // Test getValues() method
                let values = prediction.getValues()
                XCTAssertNotNil(values["class"])
                XCTAssertNotNil(values["confidence"])
                XCTAssertNotNil(values["classId"])
            }
            
            // Verify we got meaningful results
            XCTAssertGreaterThan(basePredictions.count, 0, "Should have predictions")
            if let topBasePrediction = basePredictions.first,
               let topPrediction = topBasePrediction as? RFClassificationPrediction {
                XCTAssertGreaterThan(topPrediction.confidence, 0.1, "Top prediction should have reasonable confidence")
            }
        }
    }
    
    func testBananasDetectMethod() async {
        guard let model = await TestUtils.loadBananasModel() else {
            XCTFail("Failed to load banana ripeness model")
            return
        }
        
        // Configure the model
        model.configure(threshold: 0.01, overlap: 0.0, maxObjects: 0)
        
        // Use banana image for testing
        guard let buffer = TestUtils.loadImageAsPixelBuffer(from: "Tests/assets/banana.jpg") else {
            XCTFail("Failed to load banana test image")
            return
        }
        
        // Test detect method that returns RFPrediction objects (but are actually RFClassificationPrediction)
        let (basePredictions, inferenceError) = await model.detect(pixelBuffer: buffer)
        
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
    
    func testBananasGenericDetectMethod() async {
        guard let model = await TestUtils.loadBananasModel() else {
            XCTFail("Failed to load banana ripeness model")
            return
        }
        
        // Configure the model
        model.configure(threshold: 0.01, overlap: 0.0, maxObjects: 0)
        
        // Use banana image for testing
        guard let buffer = TestUtils.loadImageAsPixelBuffer(from: "Tests/assets/banana.jpg") else {
            XCTFail("Failed to load banana test image")
            return
        }
        
        // Test generic detect method that returns RFPrediction objects
        let (predictions, inferenceError) = await model.detect(pixelBuffer: buffer)
        
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
    func testBananasUIImageClassification() async {
        guard let model = await TestUtils.loadBananasModel() else {
            XCTFail("Failed to load banana ripeness model")
            return
        }
        
        // Configure the model
        model.configure(threshold: 0.01, overlap: 0.0, maxObjects: 0)
        
        // Load UIImage from test assets
        guard let image = TestUtils.loadUIImage(from: "Tests/assets/banana.jpg") else {
            XCTFail("Failed to load banana test image as UIImage")
            return
        }
        
        // Test detect method with UIImage
        let (basePredictions, inferenceError) = await model.detect(image: image)
        
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
                XCTAssertGreaterThanOrEqual(prediction.classId, 0, "Class index should be >= 0")
            }
            
            // Verify meaningful results
            if let topBasePrediction = basePredictions.first,
               let topPrediction = topBasePrediction as? RFClassificationPrediction {
                XCTAssertGreaterThan(topPrediction.confidence, 0.1, "Top prediction should have reasonable confidence")
            }
        }
    }
    #endif
    
    // MARK: - Banana Classification Model Version 4 Tests
    
    func testLoadBananasModelV4() async {
        guard let model = await TestUtils.loadBananasModel(modelVersion: 4) else {
            XCTFail("Failed to load banana ripeness model v4")
            return
        }
        
        // Configure the model
        model.configure(threshold: 0.1, overlap: 0.0, maxObjects: 0)
        
        self.model = model
    }
    
    func testBananasClassificationInferenceV4() async {
        guard let model = await TestUtils.loadBananasModel(modelVersion: 4) else {
            XCTFail("Failed to load banana ripeness model v4")
            return
        }
        
        // Configure the model with low threshold to get more predictions
        model.configure(threshold: 0.01, overlap: 0.0, maxObjects: 0)
        
        // Use banana image for testing
        guard let buffer = TestUtils.loadImageAsPixelBuffer(from: "Tests/assets/banana.jpg") else {
            XCTFail("Failed to load banana test image")
            return
        }
        
        // Test detect method with CVPixelBuffer
        let (basePredictions, inferenceError) = await model.detect(pixelBuffer: buffer)
        
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
                XCTAssertGreaterThanOrEqual(prediction.classId, 0, "Class index should be >= 0")
                
                // Test getValues() method
                let values = prediction.getValues()
                XCTAssertNotNil(values["class"])
                XCTAssertNotNil(values["confidence"])
                XCTAssertNotNil(values["classId"])
            }
            
            // Verify we got meaningful results
            XCTAssertGreaterThan(basePredictions.count, 0, "Should have predictions")
            if let topBasePrediction = basePredictions.first,
               let topPrediction = topBasePrediction as? RFClassificationPrediction {
                XCTAssertGreaterThan(topPrediction.confidence, 0.1, "Top prediction should have reasonable confidence")
            }
        }
    }
    
    func testBananasDetectMethodV4() async {
        guard let model = await TestUtils.loadBananasModel(modelVersion: 4) else {
            XCTFail("Failed to load banana ripeness model v4")
            return
        }
        
        // Configure the model
        model.configure(threshold: 0.01, overlap: 0.0, maxObjects: 0)
        
        // Use banana image for testing
        guard let buffer = TestUtils.loadImageAsPixelBuffer(from: "Tests/assets/banana.jpg") else {
            XCTFail("Failed to load banana test image")
            return
        }
        
        // Test detect method that returns RFPrediction objects (but are actually RFClassificationPrediction)
        let (basePredictions, inferenceError) = await model.detect(pixelBuffer: buffer)
        
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
    
    func testBananasGenericDetectMethodV4() async {
        guard let model = await TestUtils.loadBananasModel(modelVersion: 4) else {
            XCTFail("Failed to load banana ripeness model v4")
            return
        }
        
        // Configure the model
        model.configure(threshold: 0.01, overlap: 0.0, maxObjects: 0)
        
        // Use banana image for testing
        guard let buffer = TestUtils.loadImageAsPixelBuffer(from: "Tests/assets/banana.jpg") else {
            XCTFail("Failed to load banana test image")
            return
        }
        
        // Test generic detect method that returns RFPrediction objects
        let (predictions, inferenceError) = await model.detect(pixelBuffer: buffer)
        
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
    func testBananasUIImageClassificationV4() async {
        guard let model = await TestUtils.loadBananasModel(modelVersion: 4) else {
            XCTFail("Failed to load banana ripeness model v4")
            return
        }
        
        // Configure the model
        model.configure(threshold: 0.01, overlap: 0.0, maxObjects: 0)
        
        // Load UIImage from test assets
        guard let image = TestUtils.loadUIImage(from: "Tests/assets/banana.jpg") else {
            XCTFail("Failed to load banana test image as UIImage")
            return
        }
        
        // Test detect method with UIImage
        let (basePredictions, inferenceError) = await model.detect(image: image)
        
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
                XCTAssertGreaterThanOrEqual(prediction.classId, 0, "Class index should be >= 0")
            }
            
            // Verify meaningful results
            if let topBasePrediction = basePredictions.first,
               let topPrediction = topBasePrediction as? RFClassificationPrediction {
                XCTAssertGreaterThan(topPrediction.confidence, 0.1, "Top prediction should have reasonable confidence")
            }
        }
    }
    #endif

    func testLoadCarsModel() async {
        guard let model = await TestUtils.loadCarsModel() else {
            XCTFail("Failed to load cars model")
            return
        }
        
        // Configure the model
        model.configure(threshold: 0.1, overlap: 0.0, maxObjects: 0)
        
        self.model = model
    }
    
    func testCarsClassificationInference() async {
        guard let model = await TestUtils.loadCarsModel() else {
            XCTFail("Failed to load cars model")
            return
        }
        
        // Configure the model with low threshold to get more predictions
        model.configure(threshold: 0.5, overlap: 0.0, maxObjects: 0)
        
        // Use cars image for testing
        guard let buffer = TestUtils.loadImageAsPixelBuffer(from: "Tests/assets/car.jpg") else {
            XCTFail("Failed to load cars test image")
            return
        }
        
        // Test detect method with CVPixelBuffer
        let (basePredictions, inferenceError) = await model.detect(pixelBuffer: buffer)
        
        XCTAssertNil(inferenceError, "Classification inference failed: \(inferenceError?.localizedDescription ?? "unknown error")")
        XCTAssertNotNil(basePredictions, "Predictions should not be nil")
        
        if let basePredictions = basePredictions {
            XCTAssertGreaterThan(basePredictions.count, 0, "Should have at least one prediction")
            XCTAssertEqual(basePredictions.count, 6, "Should have exactly 6 predictions")
            
            // Cast to RFClassificationPrediction to test specific properties
            for basePrediction in basePredictions {
                guard let prediction = basePrediction as? RFClassificationPrediction else {
                    XCTFail("Prediction should be of type RFClassificationPrediction")
                    continue
                }
                
                XCTAssertFalse(prediction.className.isEmpty, "Class name should not be empty")
                XCTAssertGreaterThanOrEqual(prediction.confidence, 0.0, "Confidence should be >= 0")
                XCTAssertLessThanOrEqual(prediction.confidence, 1.0, "Confidence should be <= 1")
                XCTAssertGreaterThanOrEqual(prediction.classId, 0, "Class index should be >= 0")
                
                // Test getValues() method
                let values = prediction.getValues()
                XCTAssertNotNil(values["class"])
                XCTAssertNotNil(values["confidence"])
                XCTAssertNotNil(values["classId"])
            }
            
            // Verify we got meaningful results
            XCTAssertGreaterThan(basePredictions.count, 0, "Should have predictions")
            if let topBasePrediction = basePredictions.first,
               let topPrediction = topBasePrediction as? RFClassificationPrediction {
                XCTAssertGreaterThan(topPrediction.confidence, 0.1, "Top prediction should have reasonable confidence")
            }
        }
    }
} 
