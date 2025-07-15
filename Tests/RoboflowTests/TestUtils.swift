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
    
    // Helper function to load local RFDetr model
    public static func loadLocalRFDetrModel() -> RFModel? {
        let rf = RoboflowMobile(apiKey: API_KEY)
        
        // Path to the local RFDetr model
        let modelPath = URL(fileURLWithPath: "Tests/assets/rfdetr_base_coreml_working_fp16.mlpackage")
        
        // Define some sample classes for COCO dataset (common for DETR models)
        let classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
        
        // Define colors for classes
        let colors: [String: String] = [
            "person": "#FF0000",
            "car": "#00FF00", 
            "bicycle": "#0000FF",
            "motorcycle": "#FFFF00",
            "bus": "#FF00FF",
            "truck": "#00FFFF"
        ]
        
        let (model, error, _, _) = rf.loadLocal(
            modelPath: modelPath,
            modelType: "detr",
            classes: classes,
            colors: colors
        )
        
        if let error = error {
            XCTFail("Failed to load local RFDetr model: \(error.localizedDescription)")
            return nil
        }
        
        XCTAssertNotNil(model, "Model should not be nil")
        guard let _ = model as? RFDetrObjectDetectionModel else {
            XCTFail("Model should be a RFDetrObjectDetectionModel")
            return nil
        }
        
        return model
    }

    // Helper function to load image and convert to CVPixelBuffer
    public static func loadImageAsPixelBuffer(from imagePath: String) -> CVPixelBuffer? {
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
    // Helper function to load UIImage for testing
    public static func loadUIImage(from imagePath: String) -> UIImage? {
        let imageURL = URL(fileURLWithPath: imagePath)
        
        guard let imageData = try? Data(contentsOf: imageURL) else {
            XCTFail("Failed to load image data from \(imagePath)")
            return nil
        }
        
        return UIImage(data: imageData)
    }
    #endif
} 