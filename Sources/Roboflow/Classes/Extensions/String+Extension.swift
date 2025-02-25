//
//  String+Extension.swift
//  Roboflow
//
//  Created by Nicholas Arner on 4/15/22.
//

import Foundation
extension String: @retroactive Error {
    
    func convertToDictionary(text: String) -> [String: Any]? {
        if let data = text.data(using: .utf8) {
            do {
                return try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any]
            } catch {
                print(error.localizedDescription)
            }
        }
        return nil
    }
    
    func toData() -> Data {
        return Data(self.utf8)
    }
}
