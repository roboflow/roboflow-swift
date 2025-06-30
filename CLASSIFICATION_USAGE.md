# ResNet Classification with Roboflow Swift

This guide explains how to use ResNet classification models with the Roboflow Swift SDK.

## Overview

The Roboflow Swift SDK now supports image classification models, including ResNet models. You can use pre-trained models for tasks like image recognition, object classification, and more.

## Classes Added

### RFClassificationPrediction
The main prediction object returned by classification models:
- `className: String` - The predicted class name
- `confidence: Float` - Confidence score (0.0 to 1.0)
- `classIndex: Int` - Index of the predicted class
- `getValues() -> [String: Any]` - Returns dictionary representation

```swift
// Example working with RFClassificationPrediction objects
let prediction = RFClassificationPrediction(className: "cat", confidence: 0.89, classIndex: 2)
print(prediction.className)     // "cat"
print(prediction.confidence)    // 0.89
print(prediction.classIndex)    // 2
print(prediction.getValues())   // ["class": "cat", "confidence": 0.89, "classIndex": 2]
```

### RFClassificationModel
The main classification model class:
- Extends `RFModel` to handle classification tasks
- Supports both local models and Roboflow API models
- Compatible with ResNet and other classification architectures
- Returns `RFClassificationPrediction` objects from `classify()` methods
- Returns `RFPrediction` objects from generic `detect()` methods (can be cast to `RFClassificationPrediction`)

## Installation

Add your ResNet `.mlmodelc` package to your iOS app bundle:

1. Download your ResNet model from the provided Google Drive link
2. Drag the `.mlmodelc` package into your Xcode project
3. Ensure "Add to target" is checked for your app target

## Usage

### Loading a Local ResNet Model

```swift
import Roboflow

let classificationModel = RFClassificationModel()

// Load model from app bundle
guard let modelURL = Bundle.main.url(forResource: "ResNet", withExtension: "mlmodelc") else {
    print("Could not find ResNet.mlmodelc in app bundle")
    return
}

if let error = classificationModel.loadLocalModel(modelPath: modelURL) {
    print("Error loading model: \(error)")
    return
}

// Configure threshold (optional)
classificationModel.configure(threshold: 0.3, overlap: 0.0, maxObjects: 0)
```

### Performing Classification

#### Using Async/Await

```swift
func classifyImage() async {
    guard let image = UIImage(named: "your_image.jpg") else { return }
    
    // Use classify() method to get RFClassificationPrediction objects directly
    let (predictions, error) = await classificationModel.classify(image: image)
    
    if let error = error {
        print("Classification error: \(error)")
        return
    }
    
    guard let predictions = predictions else {
        print("No predictions returned")
        return
    }
    
    // Work with RFClassificationPrediction objects
    for prediction in predictions {
        print("Class: \(prediction.className)")
        print("Confidence: \(String(format: "%.3f", prediction.confidence))")
        print("Class Index: \(prediction.classIndex)")
        print("Raw Values: \(prediction.getValues())")
    }
    
    // Get top prediction (predictions are sorted by confidence)
    if let topPrediction = predictions.first {
        print("🏆 Top prediction: \(topPrediction.className) (\(String(format: "%.3f", topPrediction.confidence)))")
    }
    
    // Filter high confidence predictions
    let highConfidencePredictions = predictions.filter { $0.confidence > 0.7 }
    print("High confidence predictions: \(highConfidencePredictions.count)")
}
```

#### Using Completion Handlers

```swift
func classifyImageWithCallback() {
    guard let image = UIImage(named: "your_image.jpg") else { return }
    
    // Use classify() method to get RFClassificationPrediction objects
    classificationModel.classify(image: image) { predictions, error in
        if let error = error {
            print("Classification error: \(error)")
            return
        }
        
        guard let predictions = predictions else {
            print("No predictions returned")
            return
        }
        
        // Work with RFClassificationPrediction objects
        for prediction in predictions {
            print("Class: \(prediction.className)")
            print("Confidence: \(String(format: "%.3f", prediction.confidence))")
            print("Class Index: \(prediction.classIndex)")
        }
    }
}
```

#### Using Generic detect() Method

You can also use the generic `detect()` method that returns `RFPrediction` objects:

```swift
func useGenericDetectMethod() async {
    // Convert image to CVPixelBuffer (helper method needed)
    let pixelBuffer = convertImageToPixelBuffer(image)
    
    // Use generic detect method - returns RFPrediction objects
    let (predictions, error) = await classificationModel.detect(pixelBuffer: pixelBuffer)
    
    guard let predictions = predictions else { return }
    
    for prediction in predictions {
        // Cast to RFClassificationPrediction to access specific properties
        if let classificationPrediction = prediction as? RFClassificationPrediction {
            print("Class: \(classificationPrediction.className)")
            print("Confidence: \(classificationPrediction.confidence)")
            print("Index: \(classificationPrediction.classIndex)")
        }
    }
}
```

### Loading Models from Roboflow API

If your classification model is hosted on Roboflow:

```swift
let rf = RoboflowMobile(apiKey: "your_api_key_here")

rf.load(model: "your-model-name", modelVersion: 1) { model, error, modelName, modelType in
    if let error = error {
        print("Error loading model: \(error)")
        return
    }
    
    guard let classificationModel = model as? RFClassificationModel else {
        print("Model is not a classification model")
        return
    }
    
    // Use the model...
    classificationModel.classify(image: yourImage) { predictions, error in
        // Handle results
    }
}
```

## Model Requirements

- **Input**: Images (any size, automatically resized to model requirements)
- **Format**: `.mlmodelc` (compiled Core ML models)
- **Architecture**: ResNet, EfficientNet, or other classification models
- **iOS**: Requires iOS 13.0+ for Core ML support

## Configuration Options

### Threshold
Set minimum confidence threshold for predictions:

```swift
classificationModel.configure(threshold: 0.5, overlap: 0.0, maxObjects: 0)
```

Only predictions with confidence >= threshold will be returned.

## Error Handling

Common errors and solutions:

- **"Model initialization failed"**: Check that the model file exists and is valid
- **"Unable to get classification results"**: Verify the model is a classification model
- **UnsupportedOSError**: Ensure you're running on iOS 13.0+ or macOS 10.15+

## Integration with Existing Code

The classification model is compatible with the existing Roboflow detection pipeline. You can use the same `detect()` method for compatibility, which returns `RFObjectDetectionPrediction` objects with full-image bounding boxes.

## Performance Tips

1. Use appropriate confidence thresholds to filter low-confidence predictions
2. Models are automatically cached after first load
3. Consider using background queues for model inference to avoid blocking the UI
4. Pre-load models during app startup for faster inference

## Examples

See `Examples/ResNetClassificationExample.swift` for complete usage examples.