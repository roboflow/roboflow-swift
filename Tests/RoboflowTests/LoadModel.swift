//
//  LoadModel.swift
//  
//
//  Created by Maxwell Stone on 8/10/23.
//

import XCTest
import Roboflow

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
        let (model, error, str1, str2) = await rf.load(model: "playing-cards-ow27d", modelVersion: 2)
        self.model = model
        XCTAssertNil(error)
    }
    
    func testLoadSeg() async {
        let rf = RoboflowMobile(apiKey: API_KEY)
        let (model, error, str1, str2) = await rf.load(model: "hat-1wxze-g6xvw", modelVersion: 1)
        self.model = model
        print(model)
        XCTAssertNil(error)
    }
    

}
