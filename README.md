# roboflow-swift-sdk	

<p align="center">
    </br>
    <img width="100" src="https://github.com/roboflow-ai/notebooks/raw/main/assets/roboflow_logomark_color.svg" alt="roboflow logo">
    </br>
</p>

This is the source code for the Roboflow Swift SDK. It allows you to run [Object Detection](https://blog.roboflow.com/object-detection/) models locally on your iOS device that you have trained or have been trained on [Roboflow Universe](https://universe.roboflow.com) by others. The SDK pulls down the CoreML version of the trained model and caches it lcoally for running inference on the edge. 



## Getting Started 

To get started, import Roboflow into your project: 

`import Roboflow`

and create an instance of `RoboflowMobile` that's initialzied with your API key: 

`let rf = RoboflowMobile(apiKey: API_KEY)`

You can find out how to access your API key [here](https://docs.roboflow.com/rest-api).



## Loading a CoreML Model 

Once you've initialized the SDK, you can load your model and configure it with the following code. 

```
rf.load(model: model, modelVersion: modelVersion) { [self] model, error, modelName, modelType in
    mlModel = model
    if error != nil {
        print(error?.localizedDescription as Any)
    } else {
        model?.configure(threshold: threshold, overlap: overlap, maxObjects: maxObjects)
    }
}
```



## Running Inference ## 



### Image Inference ### 

To run inference on a single image, call: 

```
mlModel.detect(image: imageToDetect) { detections, errorr in
    let detectionResults: [RFObjectDetectionPrediction] = detections!
}
```



### Video Frame Inference ###

To run inference on a video stream, you'll want to call the `detect(pixelBuffer: CVPixelBuffer, completion: **@escaping** (([RFObjectDetectionPrediction]?, Error?) -> Void))` function inside of your app's `AVCaptureVideoDataOutputSampleBufferDelegate` `captureOutput` delegate method: 

```
func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
    guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
        return
    }
    currentPixelBuffer = pixelBuffer

    mlModel?.detect(pixelBuffer: pixelBuffer, completion: { detections, error in
        if error != nil {
            print(error!)
        } else {
            let detectionResults: [RFObjectDetectionPrediction] = detections!
            ...
        }
    })
}
```

The included example app shows a complete implemention illustrating this process of setting up and running an `AVCaptureSession`. 



## Inference Results ###

You'll have noticed that when an inference is complete, the SDK returns an array of `RFObjectDetectionPrediction` results. These are structs that contain data on what object was detected in the image, as well as information on the bounding box that encapsulates that object: 

```
x: Float
y: Float
width: Float
height: Float
className: String
confdience: Float 
color: UIColor
box: CGRect
```

Call `getValues` on the returned `RFObjectDetectionPrediction` to get these results. 



## Image Uploading 

If you want to upload an image to a project for improving future versions of your model, you can do so with the `uploadImage` method. 

```
rf.uploadImage(image: image, project: project) { result in

    switch result {
        case .Success:
		print("Image uploaded successfully.")
        case .Duplicate:
        	print("You attempted to upload a duplicate image.")
        case .Error:
		print("You attempted to upload a duplicate image.")
        @unknown default:
            return
    }
}
```



## Example App ##

An example app can be found [here](https://github.com/roboflow/roboflow-swift-examples) that illusrates how to use the Roboflow SDK on an iOS app. The app uses apre-trained model hosted on Roboflow Universe for detecting the actions in a round of rock-paper-scissors. You'll have to provide your own API key. 



## Installation 

You can install the SDK either via Swift Package Manager or Cocoapods. 



#### Swift Package Manager ####

The [Swift Package Manager](https://swift.org/package-manager/) is a tool for automating the distribution of Swift code and is integrated into the `swift` compiler.

To install the Roboflow Swift SDK package into your packages, add a reference to the Roboflow Swift SDK and a targeting release version in the dependencies section in `Package.swift` file:

```
import PackageDescription

let package = Package(
    name: "YOUR_PROJECT_NAME",
    products: [],
    dependencies: [
        .package(url: "https://github.com/roboflow/swift-sdk", from: "1.0.0")
    ]
)
```

To install the package via Xcode

- Go to File -> Swift Packages -> Add Package Dependency...
- Then add https://github.com/roboflow/swift-sdk



#### Cocoapods ####

To install with Cocoapods, make sure you have Cocoapods already installed and added to your project, and then run `pod Roboflow` to your podfile: 

Then, run `pod install` in the root directory of your project. 

If you've previously installed the Roboflow SDK via Cocoapods, you'll need to update your podfile to have an entry of `pod Roboflow`. 
