//
//  ViewController.swift
//  RoboflowTestApp
//
//  Created by Nicholas Arner on 4/12/22.
//

import UIKit
import CoreML
import Vision
import AVFoundation
import Roboflow

//Obtain your API key here: https://docs.roboflow.com/rest-api
var API_KEY = "YOUR_API_KEY"

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {

    let rf = RoboflowMobile(apiKey: API_KEY)
    var mlModel: RFObjectDetectionModel!
    var bufferSize: CGSize = .zero
    var rootLayer: CALayer! = nil
    @IBOutlet weak var previewView: UIView!
    @IBOutlet weak var classLabel: UILabel!
    @IBOutlet weak var fpsLabel: UILabel!
    
    private let session = AVCaptureSession()
    private var previewLayer: AVCaptureVideoPreviewLayer! = nil
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let videoDataOutputQueue = DispatchQueue(label: "VideoDataOutput", qos: .userInitiated, attributes: [], autoreleaseFrequency: .workItem)

    private var detectionOverlay: CALayer! = nil
    var currentPixelBuffer: CVPixelBuffer!
    var start: DispatchTime!
    var end: DispatchTime!

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        
        loadRoboflowModelWith(model: "rockpaperscissors-tmlk3", modelVersion: 1, threshold: 0.1, overlap: 0.1, maxObjects: 20.0)
        checkCameraAuthorization()
    }

    func checkCameraAuthorization() {
        let authStatus = AVCaptureDevice.authorizationStatus(for: AVMediaType.video)
        
        if authStatus == AVAuthorizationStatus.denied {
            // Denied access to camera
            // Explain that we need camera access and how to change it.
            let dialog = UIAlertController(title: "Unable to access the Camera", message: "To enable access, go to Settings > Privacy > Camera and turn on Camera access for this app.", preferredStyle: UIAlertController.Style.alert)
            let okAction = UIAlertAction(title: "OK", style: UIAlertAction.Style.default, handler: nil)
            dialog.addAction(okAction)
            self.present(dialog, animated: true, completion: nil)
        } else if authStatus == AVAuthorizationStatus.notDetermined {
            // The user has not yet been presented with the option to grant access to the camera hardware.
            // Ask for it.
            AVCaptureDevice.requestAccess(for: AVMediaType.video, completionHandler: { [self] (granted) in
                if granted {
                    DispatchQueue.main.async { [self] in
                        setupAVCapture()
                    }
                }
            })
        } else {
            setupAVCapture()
        }
    }
    
    func setupAVCapture() {
        var deviceInput: AVCaptureDeviceInput!
        
        // Select a video device, make an input
        guard let videoDevice = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera], mediaType: .video, position: .back).devices.first else {
            let alert = UIAlertController(
                title: "No Camera Found",
                message: "You must run this app on a physical device with a camera.", preferredStyle: UIAlertController.Style.alert)
            alert.addAction(UIAlertAction(title: "OK", style: .cancel, handler: { (_) in
            }))
            self.present(alert, animated: true, completion: nil)
            return
        }
        do {
            deviceInput = try AVCaptureDeviceInput(device: videoDevice)
        } catch {
            print("Could not create video device input: \(error)")
            return
        }
        
        session.beginConfiguration()
        session.sessionPreset = .vga640x480
        
        // Add a video input
        guard session.canAddInput(deviceInput) else {
            print("Could not add video device input to the session")
            session.commitConfiguration()
            return
        }
        session.addInput(deviceInput)
        if session.canAddOutput(videoDataOutput) {
            session.addOutput(videoDataOutput)
            // Add a video data output
            videoDataOutput.alwaysDiscardsLateVideoFrames = true
            videoDataOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)]
            videoDataOutput.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
        } else {
            print("Could not add video data output to the session")
            session.commitConfiguration()
            return
        }
        
        let captureConnection = videoDataOutput.connection(with: .video)
        // Always process the frames
        captureConnection?.isEnabled = true
        do {
            try  videoDevice.lockForConfiguration()
            let dimensions = CMVideoFormatDescriptionGetDimensions((videoDevice.activeFormat.formatDescription))
            bufferSize.width = CGFloat(dimensions.width)
            bufferSize.height = CGFloat(dimensions.height)
            videoDevice.unlockForConfiguration()
        } catch {
            print(error)
        }
        
        session.commitConfiguration()
        
        DispatchQueue.main.async { [self] in
            previewLayer = AVCaptureVideoPreviewLayer(session: session)
            previewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
            rootLayer = previewView.layer
            previewLayer.frame = rootLayer.bounds
            rootLayer.addSublayer(previewLayer)
            rootLayer.addSublayer(classLabel.layer)
            rootLayer.addSublayer(fpsLabel.layer)
            
            setupLayers()
            updateLayerGeometry()
        }
        
        DispatchQueue.global(qos: .background).async { [self] in
            startCaptureSession()
        }
    }
    
    func setupLayers() {
        detectionOverlay = CALayer() // container layer that has all the renderings of the observations
        detectionOverlay.name = "DetectionOverlay"
        detectionOverlay.bounds = CGRect(x: 0.0,
                                         y: 0.0,
                                         width: bufferSize.width,
                                         height: bufferSize.height)
        detectionOverlay.position = CGPoint(x: rootLayer.bounds.midX, y: rootLayer.bounds.midY)
        rootLayer.addSublayer(detectionOverlay)
    }
    
    func startCaptureSession() {
        session.startRunning()
    }

    func updateLayerGeometry() {
        let bounds = rootLayer.bounds
        var scale: CGFloat

        let xScale: CGFloat = bounds.size.width / CGFloat(bufferSize.height)
        let yScale: CGFloat = bounds.size.height / CGFloat(bufferSize.width)

        scale = fmax(xScale, yScale)
        if scale.isInfinite {
            scale = 1.0
        }
        CATransaction.begin()
        CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions)

        // Rotate the layer into screen orientation and scale and mirror
        detectionOverlay.setAffineTransform(CGAffineTransform(rotationAngle: CGFloat(.pi / 2.0)).scaledBy(x: scale, y: scale))
        // Center the layer
        detectionOverlay.position = CGPoint (x: bounds.midX, y: bounds.midY)

        CATransaction.commit()
    }
    

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        currentPixelBuffer = pixelBuffer
        
        let start: DispatchTime = .now()
        
        mlModel?.detect(pixelBuffer: pixelBuffer, completion: { detections, error in
            if error != nil {
                print(error!)
            } else {
                let detectionResults: [RFObjectDetectionPrediction] = detections!
                self.determineInferenceResults(detections: detectionResults)
                
                DispatchQueue.main.async { [self] in
                    let duration = start.distance(to: .now())
                    let durationDouble = duration.toDouble()
                    var fps = 1 / durationDouble!
                    fps = round(fps)
                    fpsLabel.text = String(fps.description) + " FPS"
                }
            }
        })
    }
        
    func loadRoboflowModelWith(model: String, modelVersion: Int, threshold: Double, overlap: Double, maxObjects: Float) {
        rf.load(model: model, modelVersion: modelVersion) { [self] model, error, modelName, modelType in
            mlModel = model
            if error != nil {
                print(error?.localizedDescription as Any)
            } else {
                model?.configure(threshold: threshold, overlap: overlap, maxObjects: maxObjects)
            }
        }
    }
    
    func determineInferenceResults(detections: [RFObjectDetectionPrediction]) {
        CATransaction.begin()
        CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions)
        detectionOverlay.sublayers = nil // remove all the old recognized objects

        for detection in detections {
            let detectionInfo = detection.getValues()
            
            guard let detectedValue = detectionInfo["class"] as? String else {
                return
            }
            
            guard let x = detectionInfo["x"] as? Float else {
                return
            }

            guard let y = detectionInfo["y"] as? Float else {
                return
            }
            
            guard let width = detectionInfo["width"] as? Float else {
                return
            }

            guard let height = detectionInfo["height"] as? Float else {
                return
            }
            
            let boundingBoxColor = UIColor.purple
            let bounds = detectionOverlay.bounds
            let xs = bounds.width/bufferSize.width
            let ys = bounds.height/bufferSize.height
            let boundingBox: CGRect = CGRect(x: CGFloat(x)*xs, y: CGFloat(y)*ys, width: CGFloat(width)*xs, height: CGFloat(height)*ys)
            drawBoundingBox(boundingBox: boundingBox, color: boundingBoxColor)
            
            DispatchQueue.main.async {
                self.classLabel.text = detectedValue
            }
        }

        CATransaction.commit()
    }
    
    func drawBoundingBox(boundingBox: CGRect, color: UIColor) {
        let shapeLayer = self.createRoundedRectLayerWithBounds(boundingBox, color: color)
        detectionOverlay.addSublayer(shapeLayer)
        self.updateLayerGeometry()
    }
    
    func createRoundedRectLayerWithBounds(_ bounds: CGRect, color: UIColor) -> CALayer {
        let shapeLayer = CALayer()
        shapeLayer.bounds = bounds
        shapeLayer.position = CGPoint(x: bounds.origin.x, y: bounds.origin.y)
        shapeLayer.name = "Found Object"
        
        var colorComponents = color.cgColor.components
        colorComponents?.removeLast()
        colorComponents?.append(0.4)
        shapeLayer.backgroundColor = CGColor(colorSpace: CGColorSpaceCreateDeviceRGB(), components: colorComponents!)
        shapeLayer.cornerRadius = 7
        return shapeLayer
    }
    
}
