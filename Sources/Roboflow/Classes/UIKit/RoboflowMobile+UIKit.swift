//
//  RoboflowMobile+UIKit.swift
//  roboflow-swift
//
//  Created by Maxwell Stone on 6/12/25.
//

#if canImport(UIKit)
import UIKit

extension RoboflowMobile {
    //Upload an image to a provided project
    public func uploadImage(image: UIImage, project: String, completion: @escaping (UploadResult)->()) {
        let encodedImage = convertImageToBase64String(img: image)
        let uuid = UUID().uuidString
        
        var request = URLRequest(url: URL(string: "https://api.roboflow.com/dataset/\(project)/upload?api_key=\(apiKey!)&name=\(uuid)&split=train")!,timeoutInterval: Double.infinity)

        request.addValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        request.httpMethod = "POST"
        request.httpBody = encodedImage.toData()
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            // Parse Response to String
            guard let data = data else {
                completion(UploadResult.Error)
                return
            }

            do {
                let dict = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any]
                let duplicate = dict!["duplicate"] as? Bool
                
                if duplicate ==  true {
                    completion(UploadResult.Duplicate)
                } else {
                    let success = dict!["success"] as! Bool
                    if success == true {
                        completion(UploadResult.Success)
                    } else {
                        completion(UploadResult.Error)
                    }
                }

            } catch {
                print(error.localizedDescription)
                completion(UploadResult.Error)
            }
        }.resume()
    }
    
    func convertImageToBase64String (img: UIImage) -> String {
        return img.jpegData(compressionQuality: 1)?.base64EncodedString() ?? ""
    }
}

#endif
