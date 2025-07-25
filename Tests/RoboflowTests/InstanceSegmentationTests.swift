//
//  InstanceSegmentationTests.swift
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

final class InstanceSegmentationTests: XCTestCase {
    var model: RFModel?

    override func setUpWithError() throws {
        // Put setup code here. This method is called before the invocation of each test method in the class.
        let rf = RoboflowMobile(apiKey: API_KEY)
        rf.clearModelCache(modelName: "hat-1wxze-g6xvw", modelVersion: 1)
    }

    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }
    
    // MARK: - Instance Segmentation Model Tests
    
    func testLoadSeg() async {
        let rf = RoboflowMobile(apiKey: API_KEY)
        let (model, error, _, _) = await rf.load(model: "hat-1wxze-g6xvw", modelVersion: 1)
        self.model = model
        XCTAssertNil(error)
        XCTAssertNotNil(model)
    }
    
    // test running inference on segmentation model
    func testRunSegmentationInference() async {
        guard let model = await TestUtils.loadInstanceSegmentationModel() else {
            XCTFail("Failed to load instance segmentation model")
            return
        }

        guard let buffer = TestUtils.loadImageAsPixelBuffer(from: "Tests/assets/cap.jpg") else {
            XCTFail("Failed to load cap test image")
            return
        }
        
        let (results, inferenceError) = await model.detect(pixelBuffer: buffer)
        XCTAssertNil(inferenceError)
        // Note: predictions might be nil if no objects are detected in the test image, which is expected
        XCTAssertNotNil(results)
        XCTAssert(results?.count ?? 0 > 0)
    }
    
    func testInstanceSegmentationInference() async {
        guard let model = await TestUtils.loadInstanceSegmentationModel() else {
            XCTFail("Failed to load instance segmentation model")
            return
        }

        guard let buffer = TestUtils.loadImageAsPixelBuffer(from: "Tests/assets/cap.jpg") else {
            XCTFail("Failed to load cap test image")
            return
        }
        
        let (predictions, inferenceError) = await model.detect(pixelBuffer: buffer)
        XCTAssertNil(inferenceError, "Instance segmentation inference failed: \(inferenceError?.localizedDescription ?? "unknown error")")
        XCTAssertNotNil(predictions, "Predictions should not be nil")
        
        if let predictions = predictions {
            XCTAssertGreaterThan(predictions.count, 0, "Should have at least one prediction")
            
            // Cast to RFInstanceSegmentationPrediction to test specific properties
            for prediction in predictions {
                guard let segPrediction = prediction as? RFInstanceSegmentationPrediction else {
                    XCTFail("Prediction should be of type RFInstanceSegmentationPrediction")
                    continue
                }
                
                XCTAssertFalse(segPrediction.className.isEmpty, "Class name should not be empty")
                XCTAssertGreaterThanOrEqual(segPrediction.confidence, 0.0, "Confidence should be >= 0")
                XCTAssertLessThanOrEqual(segPrediction.confidence, 1.0, "Confidence should be <= 1")
                
                // Test bounding box properties
                XCTAssertGreaterThan(segPrediction.width, 0, "Width should be > 0")
                XCTAssertGreaterThan(segPrediction.height, 0, "Height should be > 0")
                
                // Test getValues() method
                let values = segPrediction.getValues()
                XCTAssertNotNil(values["class"])
                XCTAssertNotNil(values["confidence"])
                XCTAssertNotNil(values["x"])
                XCTAssertNotNil(values["y"])
                XCTAssertNotNil(values["width"])
                XCTAssertNotNil(values["height"])
                XCTAssertNotNil(values["points"])
            }
        }
    }
    
    #if canImport(UIKit)
    func testInstanceSegmentationUIImageInference() async {
        guard let model = await TestUtils.loadInstanceSegmentationModel() else {
            XCTFail("Failed to load instance segmentation model")
            return
        }
        
        // Load UIImage from test assets
        guard let image = TestUtils.loadUIImage(from: "Tests/assets/cap.jpg") else {
            XCTFail("Failed to load cap test image as UIImage")
            return
        }
        
        // Test detect method with UIImage
        let (predictions, inferenceError) = await model.detect(image: image)
        
        XCTAssertNil(inferenceError, "UIImage instance segmentation inference failed: \(inferenceError?.localizedDescription ?? "unknown error")")
        XCTAssertNotNil(predictions, "Predictions should not be nil")
        
        if let predictions = predictions {
            XCTAssertGreaterThan(predictions.count, 0, "Should have at least one prediction")
            
            // Test RFInstanceSegmentationPrediction properties by casting
            for prediction in predictions {
                guard let segPrediction = prediction as? RFInstanceSegmentationPrediction else {
                    XCTFail("Prediction should be of type RFInstanceSegmentationPrediction")
                    continue
                }
                
                XCTAssertFalse(segPrediction.className.isEmpty, "Class name should not be empty")
                XCTAssertGreaterThanOrEqual(segPrediction.confidence, 0.0, "Confidence should be >= 0")
                XCTAssertLessThanOrEqual(segPrediction.confidence, 1.0, "Confidence should be <= 1")
                XCTAssertNotNil(segPrediction.points)
            }
            
            // Verify meaningful results
            if let firstPrediction = predictions.first,
               let segPrediction = firstPrediction as? RFInstanceSegmentationPrediction {
                XCTAssertGreaterThan(segPrediction.confidence, 0.1, "Top prediction should have reasonable confidence")
            }
        }
    }
    #endif
} 