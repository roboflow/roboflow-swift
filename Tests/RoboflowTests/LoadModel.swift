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

// cash counter api_key (already public)
var API_KEY = "fEto4us79wdzRJ2jkO6U"

final class LoadModel: XCTestCase {
    var model: RFModel?

    override func setUpWithError() throws {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
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

        // Load the hard-hat.jpeg image and convert to CVPixelBuffer
        let imagePath = "Tests/assets/hard-hat.jpeg"
        let imageURL = URL(fileURLWithPath: imagePath)
        
        guard let imageSource = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) else {
            XCTFail("Failed to load test image from \(imagePath)")
            return
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
        
        guard status == kCVReturnSuccess, 
              let buffer = pixelBuffer,
              let unwrappedModel = model else {
            XCTFail("Failed to create pixel buffer or model is nil")
            return
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
            return
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        CVPixelBufferUnlockBaseAddress(buffer, [])
        
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

        // Load the cap.jpg image and convert to CVPixelBuffer
        let imagePath = "Tests/assets/cap.jpg"
        let imageURL = URL(fileURLWithPath: imagePath)
        
        guard let imageSource = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) else {
            XCTFail("Failed to load test image from \(imagePath)")
            return
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
        
        guard status == kCVReturnSuccess, 
              let buffer = pixelBuffer,
              let unwrappedModel = model else {
            XCTFail("Failed to create pixel buffer or model is nil")
            return
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
            return
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        CVPixelBufferUnlockBaseAddress(buffer, [])
        
        let (results, inferenceError) = await unwrappedModel.detect(pixelBuffer: buffer)
        XCTAssertNil(inferenceError)
        // Note: predictions might be nil if no objects are detected in the test image, which is expected
        XCTAssertNotNil(results)
        XCTAssert(results?.count ?? 0 > 0)
    }
    

}
