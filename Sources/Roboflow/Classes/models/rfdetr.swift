//
//  rfdetr.swift
//  Roboflow
//
//  Created by AI Assistant
//

import CoreML

/// Model Prediction Input Type
class RFDetrInput : MLFeatureProvider {

    /// image_input as MultiArray (Float16, 1 × 3 × 560 × 560)
    var image_input: MLMultiArray

    var featureNames: Set<String> {
        get {
            return ["image_input"]
        }
    }
    
    func featureValue(for featureName: String) -> MLFeatureValue? {
        if (featureName == "image_input") {
            return MLFeatureValue(multiArray: image_input)
        }
        return nil
    }
    
    init(image_input: MLMultiArray) {
        self.image_input = image_input
    }
}

/// Model Prediction Output Type
class RFDetrOutput : MLFeatureProvider {

    /// Source provided by CoreML
    private let provider : MLFeatureProvider

    /// Bounding boxes as multidimensional array of doubles
    lazy var boxes: MLMultiArray = {
        [unowned self] in return self.provider.featureValue(for: "boxes")!.multiArrayValue
    }()!

    /// Bounding boxes as multidimensional array of doubles
    @available(macOS 12.0, iOS 15.0, tvOS 15.0, watchOS 8.0, *)
    var boxesShapedArray: MLShapedArray<Double> {
        return MLShapedArray<Double>(self.boxes)
    }

    /// Confidence scores as multidimensional array of doubles
    lazy var scores: MLMultiArray = {
        [unowned self] in return self.provider.featureValue(for: "scores")!.multiArrayValue
    }()!

    /// Confidence scores as multidimensional array of doubles
    @available(macOS 12.0, iOS 15.0, tvOS 15.0, watchOS 8.0, *)
    var scoresShapedArray: MLShapedArray<Double> {
        return MLShapedArray<Double>(self.scores)
    }

    /// Class labels as multidimensional array of integers
    lazy var labels: MLMultiArray = {
        [unowned self] in return self.provider.featureValue(for: "labels")!.multiArrayValue
    }()!

    /// Class labels as multidimensional array of integers
    @available(macOS 12.0, iOS 15.0, tvOS 15.0, watchOS 8.0, *)
    var labelsShapedArray: MLShapedArray<Double> {
        return MLShapedArray<Double>(self.labels)
    }

    var featureNames: Set<String> {
        return self.provider.featureNames
    }
    
    func featureValue(for featureName: String) -> MLFeatureValue? {
        return self.provider.featureValue(for: featureName)
    }

    init(boxes: MLMultiArray, scores: MLMultiArray, labels: MLMultiArray) {
        self.provider = try! MLDictionaryFeatureProvider(dictionary: [
            "boxes" : MLFeatureValue(multiArray: boxes),
            "scores" : MLFeatureValue(multiArray: scores),
            "labels" : MLFeatureValue(multiArray: labels)
        ])
    }

    init(features: MLFeatureProvider) {
        self.provider = features
    }
}

/// Class for model loading and prediction
class RFDetr {
    let model: MLModel

    /// URL of model assuming it was installed in the same bundle as this class
    class var urlOfModelInThisBundle : URL {
        let bundle = Bundle(for: self)
        return bundle.url(forResource: "rfdetr", withExtension:"mlmodelc")!
    }

    /**
        Construct RFDetr instance with an existing MLModel object.

        Usually the application does not use this initializer unless it makes a subclass of RFDetr.
        Such application may want to use `MLModel(contentsOfURL:configuration:)` and `RFDetr.urlOfModelInThisBundle` to create a MLModel object to pass-in.

        - parameters:
          - model: MLModel object
    */
    init(model: MLModel) {
        self.model = model
    }

    /**
        Construct RFDetr instance by automatically loading the model from the app's bundle.
    */
    @available(*, deprecated, message: "Use init(configuration:) instead and handle errors appropriately.")
    @available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
    convenience init() {
        try! self.init(contentsOf: type(of:self).urlOfModelInThisBundle)
    }

    /**
        Construct a model with configuration

        - parameters:
           - configuration: the desired model configuration

        - throws: an NSError object that describes the problem
    */
    @available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
    convenience init(configuration: MLModelConfiguration) throws {
        try self.init(contentsOf: type(of:self).urlOfModelInThisBundle, configuration: configuration)
    }

    /**
        Construct RFDetr instance with explicit path to mlmodelc file
        - parameters:
           - modelURL: the file url of the model

        - throws: an NSError object that describes the problem
    */
    @available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
    convenience init(contentsOf modelURL: URL) throws {
        try self.init(model: MLModel(contentsOf: modelURL))
    }

    /**
        Construct a model with URL of the .mlmodelc directory and configuration

        - parameters:
           - modelURL: the file url of the model
           - configuration: the desired model configuration

        - throws: an NSError object that describes the problem
    */
    @available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
    convenience init(contentsOf modelURL: URL, configuration: MLModelConfiguration) throws {
        try self.init(model: MLModel(contentsOf: modelURL, configuration: configuration))
    }

    /**
        Construct RFDetr instance asynchronously with optional configuration.

        Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

        - parameters:
          - configuration: the desired model configuration
          - handler: the completion handler to be called when the model loading completes successfully or unsuccessfully
    */
    @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
    class func load(configuration: MLModelConfiguration = MLModelConfiguration(), completionHandler handler: @escaping (Swift.Result<RFDetr, Error>) -> Void) {
        return self.load(contentsOf: self.urlOfModelInThisBundle, configuration: configuration, completionHandler: handler)
    }

    /**
        Construct RFDetr instance asynchronously with optional configuration.

        Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

        - parameters:
          - configuration: the desired model configuration
    */
    @available(macOS 12.0, iOS 15.0, tvOS 15.0, watchOS 8.0, *)
    class func load(configuration: MLModelConfiguration = MLModelConfiguration()) async throws -> RFDetr {
        return try await self.load(contentsOf: self.urlOfModelInThisBundle, configuration: configuration)
    }

    /**
        Construct RFDetr instance asynchronously with URL of the .mlmodelc directory with optional configuration.

        Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

        - parameters:
          - modelURL: the URL to the model
          - configuration: the desired model configuration
          - handler: the completion handler to be called when the model loading completes successfully or unsuccessfully
    */
    @available(macOS 11.0, iOS 14.0, tvOS 14.0, watchOS 7.0, *)
    class func load(contentsOf modelURL: URL, configuration: MLModelConfiguration = MLModelConfiguration(), completionHandler handler: @escaping (Swift.Result<RFDetr, Error>) -> Void) {
        MLModel.load(contentsOf: modelURL, configuration: configuration) { result in
            switch result {
            case .failure(let error):
                handler(.failure(error))
            case .success(let model):
                handler(.success(RFDetr(model: model)))
            }
        }
    }

    /**
        Construct RFDetr instance asynchronously with URL of the .mlmodelc directory with optional configuration.

        Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

        - parameters:
          - modelURL: the URL to the model
          - configuration: the desired model configuration
    */
    @available(macOS 12.0, iOS 15.0, tvOS 15.0, watchOS 8.0, *)
    class func load(contentsOf modelURL: URL, configuration: MLModelConfiguration = MLModelConfiguration()) async throws -> RFDetr {
        let model = try await MLModel.load(contentsOf: modelURL, configuration: configuration)
        return RFDetr(model: model)
    }

    /**
        Make a prediction using the structured interface

        - parameters:
           - input: the input to the prediction as RFDetrInput

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as RFDetrOutput
    */
    func prediction(input: RFDetrInput) throws -> RFDetrOutput {
        return try self.prediction(input: input, options: MLPredictionOptions())
    }

    /**
        Make a prediction using the structured interface

        - parameters:
           - input: the input to the prediction as RFDetrInput
           - options: prediction options 

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as RFDetrOutput
    */
    func prediction(input: RFDetrInput, options: MLPredictionOptions) throws -> RFDetrOutput {
        let outFeatures = try model.prediction(from: input, options:options)
        return RFDetrOutput(features: outFeatures)
    }

    /**
        Make a prediction using the convenience interface

        - parameters:
            - pixelBuffer: input image as CVPixelBuffer

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as RFDetrOutput
    */
    // func prediction(pixelBuffer: CVPixelBuffer) throws -> RFDetrOutput {
    //     let input_ = try RFDetrInput(pixelBuffer: pixelBuffer)
    //     return try self.prediction(input: input_)
    // }

    /**
        Make a batch prediction using the structured interface

        - parameters:
           - inputs: the inputs to the prediction as [RFDetrInput]
           - options: prediction options 

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as [RFDetrOutput]
    */
    @available(macOS 10.15, iOS 13.0, tvOS 13.0, watchOS 6.0, *)
    func predictions(inputs: [RFDetrInput], options: MLPredictionOptions = MLPredictionOptions()) throws -> [RFDetrOutput] {
        let batchIn = MLArrayBatchProvider(array: inputs)
        let batchOut = try model.predictions(from: batchIn, options: options)
        var results : [RFDetrOutput] = []
        results.reserveCapacity(inputs.count)
        for i in 0..<batchOut.count {
            let outProvider = batchOut.features(at: i)
            let result =  RFDetrOutput(features: outProvider)
            results.append(result)
        }
        return results
    }
} 