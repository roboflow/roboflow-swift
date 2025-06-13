//
//  Roboflow.swift
//  Roboflow
//
//  Created by Nicholas Arner on 4/12/22.
//

import CoreML
import Vision

///Interface for interacting with the Roboflow API
@available(macOS 10.15, *)
public class RoboflowMobile: NSObject {
    
    var apiKey: String!
    var deviceID: String!
    private var retries = 2
    
    //Initalize the SDK with the user's authorization key
    @available(macOS 12.0, *)
    public init (apiKey: String) {
        super.init()
        self.apiKey = apiKey
        
        //Generate a unique device ID
        guard let deviceID = getDeviceId() else {
            fatalError("Failed to generate device ID")
        }
        
        self.deviceID = deviceID
    }
    
    func getModelClass(modelType: String) -> RFModel {
        if (modelType.contains("seg")) {
            return RFInstanceSegmentationModel()
        }
        return RFObjectDetectionModel()
    }
    
    //Start the process of fetching the CoreMLModel
    @available(*, renamed: "load(model:modelVersion:)")
    public func load(model: String, modelVersion: Int, completion: @escaping (RFModel?, Error?, String, String)->()) {
        if let modelInfo = loadModelCache(modelName: model, modelVersion: modelVersion),
            let modelURL = modelInfo["compiledModelURL"] as? String,
            let colors = modelInfo["colors"] as? [String: String],
           let classes = modelInfo["classes"] as? [String],
            let name = modelInfo["name"] as? String,
            let modelType = modelInfo["modelType"] as? String {
            
            getConfigDataBackground(modelName: model, modelVersion: modelVersion, apiKey: apiKey, deviceID: deviceID)
            
            let modelObject = getModelClass(modelType: modelType)

            do {
                let documentsURL = try FileManager.default.url(for: .documentDirectory,
                                                                in: .userDomainMask,
                                                                appropriateFor: nil,
                                                                create: false)
                _ = modelObject.loadMLModel(modelPath: documentsURL.appendingPathComponent(modelURL), colors: colors, classes: classes)
                
                completion(modelObject, nil, name, modelType)
            } catch {
                clearAndRetryLoadingModel(model, modelVersion, completion)
            }
        } else if retries > 0 {
            clearModelCache(modelName: model, modelVersion: modelVersion)
            retries -= 1
            getModelData(modelName: model, modelVersion: modelVersion, apiKey: apiKey, deviceID: deviceID) { [self] fetchedModel, error, modelName, modelType, colors, classes in
                if let err = error {
                    completion(nil, err, "", "")
                } else if let fetchedModel = fetchedModel {
                    let modelObject = getModelClass(modelType: modelType)
                    _ = modelObject.loadMLModel(modelPath: fetchedModel, colors: colors ?? [:], classes: classes ?? [])
                    completion(modelObject, nil, modelName, modelType)
                } else {
                    print("No Model Found. Trying Again.")
                    clearAndRetryLoadingModel(model, modelVersion, completion)
                }
            }
        } else {
            print("Error Loading Model. Check your API_KEY, project name, and version along with your network connection.")
        }
    }

    private func clearAndRetryLoadingModel(_ model: String, _ modelVersion: Int, _ completion: @escaping (RFModel?, Error?, String, String)->()) {
        clearModelCache(modelName: model, modelVersion: modelVersion)
        self.load(model: model, modelVersion: modelVersion, completion: completion)
    }

    public func load(model: String, modelVersion: Int) async -> (RFModel?, Error?, String, String) {
        return await withCheckedContinuation { continuation in
            load(model: model, modelVersion: modelVersion) { result1, result2, result3, result4 in
                continuation.resume(returning: (result1, result2, result3, result4))
            }
        }
    }
    
    func getConfigData(modelName: String, modelVersion: Int, apiKey: String, deviceID: String, completion: @escaping (([String: Any]?, Error?) -> Void)) {
        var request = URLRequest(url: URL(string: "https://api.roboflow.com/coreml/\(modelName)/\(String(modelVersion))?api_key=\(apiKey)&device=\(deviceID)&bundle=\(Bundle.main.bundleIdentifier ?? "nobundle")")!,timeoutInterval: Double.infinity)
        request.addValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        request.httpMethod = "GET"
        
        // Execute Post Request
        URLSession.shared.dataTask(with: request, completionHandler: { data, response, error in
            
            // Parse Response to String
            guard let data = data else {
                completion(nil, error)
                return
            }
            
            // Convert Response String to Dictionary
            do {
                let dict = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any]
                completion(dict, nil)
                
            } catch {
                print(error.localizedDescription)
                completion(nil, error.localizedDescription)
            }
        }).resume()
    }
    
    private func getConfigDataBackground(modelName: String, modelVersion: Int, apiKey: String, deviceID: String) {
        DispatchQueue.global(qos: .background).async {
            self.getConfigData(modelName: modelName, modelVersion: modelVersion, apiKey: apiKey, deviceID: deviceID, completion: {_,_ in })
        }
    }
    
    
    //Get the model metadata from the Roboflow API
    private func getModelData(modelName: String, modelVersion: Int, apiKey: String, deviceID: String, completion: @escaping (URL?, Error?, String, String, [String: String]?, [String]?)->()) {
        getConfigData(modelName: modelName, modelVersion: modelVersion, apiKey: apiKey, deviceID: deviceID) { data, error in
            if let error = error {
                completion(nil, error, "", "", nil, nil)
                return
            }
            
            guard let data = data,
                  let coreMLDict = data["coreml"] as? [String: Any],
                  let name = coreMLDict["name"] as? String,
                  let modelType = coreMLDict["modelType"] as? String,
                  let modelURLString = coreMLDict["model"] as? String,
                  let modelURL = URL(string: modelURLString) else {
                completion(nil, error, "", "", nil, nil)
                return
            }
            
            let colors = coreMLDict["colors"] as? [String: String]
            let classes = coreMLDict["classes"] as? [String]
            
            
            //Download the model from the link in the API response
            self.downloadModelFile(modelName: "\(modelName)-\(modelVersion).mlmodel", modelVersion: modelVersion, modelURL: modelURL) { fetchedModel, error in
                if let error = error {
                    completion(nil, error, "", "", nil, nil)
                    return
                }
                
                if let fetchedModel = fetchedModel {
                    _ = self.cacheModelInfo(modelName: modelName, modelVersion: modelVersion, colors: colors ?? [:], classes: classes ?? [], name: name, modelType: modelType, compiledModelURL: fetchedModel)
                    completion(fetchedModel, nil, name, modelType, colors, classes)
                } else {
                    completion(nil, error, "", "", nil, nil)
                }
            }
        }
    }


    private func cacheModelInfo(modelName: String, modelVersion: Int, colors: [String: String], classes: [String], name: String, modelType: String, compiledModelURL: URL) -> [String: Any]? {
        let modelInfo: [String : Any] = [
            "colors": colors,
            "classes": classes,
            "name": name,
            "modelType": modelType,
            "compiledModelURL": compiledModelURL.lastPathComponent
        ]
        
        do {
            let encodedData = try NSKeyedArchiver.archivedData(withRootObject: modelInfo, requiringSecureCoding: true)
            UserDefaults.standard.set(encodedData, forKey: "\(modelName)-\(modelVersion)")
            return modelInfo
        } catch {
            print("Error while caching model info: \(error.localizedDescription)")
            return nil
        }
    }
        
    private func loadModelCache(modelName: String, modelVersion: Int) -> [String: Any]? {
        do {
            if let modelInfoData = UserDefaults.standard.data(forKey: "\(modelName)-\(modelVersion)") {
                let decodedData = try NSKeyedUnarchiver.unarchivedObject(ofClasses: [NSDictionary.self, NSString.self, NSArray.self], from: modelInfoData) as? [String: Any]
                return decodedData
            } else {
                print("Error: Could not find data for key \(modelName)-\(modelVersion)")
            }
        } catch {
            print("Error unarchiving data: \(error.localizedDescription)")
        }
        return nil
    }

    public func clearModelCache(modelName: String, modelVersion: Int) {
        UserDefaults.standard.removeObject(forKey: "\(modelName)-\(modelVersion)")
    }
    
    //Download the model link with the provided URL from the Roboflow API
    private func downloadModelFile(modelName: String, modelVersion: Int, modelURL: URL, completion: @escaping (URL?, Error?)->()) {
        
        downloadModel(signedURL: modelURL) { url in
            if url != nil {
                do {
                    //Compile the downloaded model
                    let compiledModelURL = try MLModel.compileModel(at: url!)
                    let documentsURL = try
                    FileManager.default.url(for: .documentDirectory,
                                            in: .userDomainMask,
                                            appropriateFor: nil,
                                            create: false)
                    let savedURL = documentsURL.appendingPathComponent("\(modelName)-\(modelVersion).mlmodelc")
                    do {
                        try FileManager.default.moveItem(at: compiledModelURL, to: savedURL)
                    } catch {
                        print(error.localizedDescription)
                    }
                    completion(savedURL, nil)
                } catch {
                    print(error.localizedDescription)
                    completion(nil, error)
                }
            }
        }
    }
    
    func downloadModel(signedURL: URL, completion: @escaping ((URL?) -> Void)) {
        let downloadTask = URLSession.shared.downloadTask(with: signedURL) {
            urlOrNil, responseOrNil, errorOrNil in
            guard let fileURL = urlOrNil else {
                completion(nil)
                return
            }
            completion(fileURL)
        }
        downloadTask.resume()
    }
}

