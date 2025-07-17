//
//  TestUtils.swift
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

// API Keys
public let API_KEY = "rf_EsVTlbAbaZPLmAFuQwWoJgFpMU82" // cash counter api_key (already public); yolo models
public let BANANA_API_KEY = "rf_EsVTlbAbaZPLmAFuQwWoJgFpMU82" // banana ripeness staging api_key; classification models

public class TestUtils {
    // Helper function to get resource URL
    private static func getResourceURL(for filename: String, withExtension ext: String) -> URL? {
        // First try to get the test bundle for Swift Package Manager
        var bundle: Bundle
        
        // Check if Bundle.module is available (Swift 5.3+)
        if #available(macOS 10.15, iOS 13.0, *) {
            #if SWIFT_PACKAGE && swift(>=5.3)
            if let moduleBundle = Bundle.module as Bundle? {
                bundle = moduleBundle
            } else {
                bundle = Bundle(for: TestUtils.self)
            }
            #else
            bundle = Bundle(for: TestUtils.self)
            #endif
        } else {
            bundle = Bundle(for: TestUtils.self)
        }
        
        // Try to find in the bundle first
        if let url = bundle.url(forResource: filename, withExtension: ext) {
            return url
        }
        
        // Try looking in the assets subdirectory in bundle
        if let url = bundle.url(forResource: filename, withExtension: ext, subdirectory: "assets") {
            return url
        }
        
        // For xcodebuild tests, try common relative paths from the working directory
        let fileManager = FileManager.default
        let currentDir = fileManager.currentDirectoryPath
        
        let commonPaths = [
            // Common locations for xcodebuild
            "\(currentDir)/Tests/assets/\(filename).\(ext)",
            "\(currentDir)/tests/assets/\(filename).\(ext)",
            "\(currentDir)/.swiftpm/xcode/Tests/assets/\(filename).\(ext)",
            // Fallback paths
            "Tests/assets/\(filename).\(ext)",
            "tests/assets/\(filename).\(ext)", 
            "assets/\(filename).\(ext)",
            "\(filename).\(ext)"
        ]
        
        for path in commonPaths {
            if fileManager.fileExists(atPath: path) {
                return URL(fileURLWithPath: path)
            }
        }
        
        // Debug: print current directory and search paths
        print("Current directory: \(currentDir)")
        print("Searched for: \(filename).\(ext)")
        print("Searched paths:")
        for path in commonPaths {
            print("  - \(path) (exists: \(fileManager.fileExists(atPath: path)))")
        }
        
        return nil
    }
    
    // Helper function to load remote banana classification model
    public static func loadBananasModel(modelVersion: Int = 6) async -> RFModel? {
        let rf = RoboflowMobile(apiKey: BANANA_API_KEY, apiURL: "https://api.roboflow.com")
        let (model, error, _, _) = await rf.load(model: "banana-ripeness-frqdw", modelVersion: modelVersion)
        
        if let error = error {
            XCTFail("Failed to load banana ripeness model: \(error.localizedDescription)")
            return nil
        }
        
        XCTAssertNotNil(model, "Model should not be nil")
        guard let _ = model as? RFClassificationModel else {
            XCTFail("Model should be a classification model")
            return nil
        }
        return model
    }

    public static func loadCarsModel(modelVersion: Int = 1) async -> RFModel? {
        let rf = RoboflowMobile(apiKey: API_KEY)
        let (model, error, _, _) = await rf.load(model: "multiclass-gyn4p-l5m6c", modelVersion: modelVersion)
        
        if let error = error {
            XCTFail("Failed to load cars model: \(error.localizedDescription)")
            return nil
        }
        
        XCTAssertNotNil(model, "Model should not be nil")
        guard let _ = model as? RFClassificationModel else {
            XCTFail("Model should be a classification model")
            return nil
        }
        return model
    }
    
    // Helper function to load object detection model
    public static func loadObjectDetectionModel() async -> RFModel? {
        let rf = RoboflowMobile(apiKey: API_KEY)
        let (model, error, _, _) = await rf.load(model: "hard-hat-sample-txcpu", modelVersion: 6)
        
        if let error = error {
            XCTFail("Failed to load object detection model: \(error.localizedDescription)")
            return nil
        }
        
        XCTAssertNotNil(model, "Model should not be nil")
        return model
    }
    
    // Helper function to load instance segmentation model
    public static func loadInstanceSegmentationModel() async -> RFModel? {
        let rf = RoboflowMobile(apiKey: API_KEY)
        let (model, error, _, _) = await rf.load(model: "hat-1wxze-g6xvw", modelVersion: 1)
        
        if let error = error {
            XCTFail("Failed to load instance segmentation model: \(error.localizedDescription)")
            return nil
        }
        
        XCTAssertNotNil(model, "Model should not be nil")
        return model
    }
    
    // Helper function to load RFDetr model from API
    public static func loadRFDetrModel() async -> RFModel? {
        let rf = RoboflowMobile(apiKey: API_KEY)
        let (model, error, _, _) = await rf.load(model: "hard-hat-sample-txcpu", modelVersion: 7)
        
        if let error = error {
            XCTFail("Failed to load RFDetr model: \(error.localizedDescription)")
            return nil
        }
        
        XCTAssertNotNil(model, "RFDetr model should not be nil")
        return model
    }

    // Helper function to load image and convert to CVPixelBuffer
    public static func loadImageAsPixelBuffer(from imagePath: String) -> CVPixelBuffer? {
        // Extract filename and extension from the path
        let url = URL(fileURLWithPath: imagePath)
        let filename = url.deletingPathExtension().lastPathComponent
        let ext = url.pathExtension
        
        guard let imageURL = getResourceURL(for: filename, withExtension: ext) else {
            XCTFail("Failed to find test image: \(imagePath). Searched in bundle and common locations.")
            return nil
        }
        
        guard let imageSource = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) else {
            XCTFail("Failed to load test image from \(imageURL.path)")
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
    // Helper function to load UIImage for testing
    public static func loadUIImage(from imagePath: String) -> UIImage? {
        // Extract filename and extension from the path
        let url = URL(fileURLWithPath: imagePath)
        let filename = url.deletingPathExtension().lastPathComponent
        let ext = url.pathExtension
        
        guard let imageURL = getResourceURL(for: filename, withExtension: ext) else {
            XCTFail("Failed to find test image: \(imagePath). Searched in bundle and common locations.")
            return nil
        }
        
        guard let imageData = try? Data(contentsOf: imageURL) else {
            XCTFail("Failed to load image data from \(imageURL.path)")
            return nil
        }
        
        return UIImage(data: imageData)
    }
    #endif
} 