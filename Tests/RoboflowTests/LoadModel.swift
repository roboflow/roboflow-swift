//
//  LoadModel.swift
//  
//
//  Created by Maxwell Stone on 8/10/23.
//

import XCTest
import Roboflow
import CoreVideo
import CoreGraphics
import ImageIO
import Foundation

// Legacy test file - Tests have been moved to separate files:
// - ClassificationTests.swift
// - ObjectDetectionTests.swift 
// - InstanceSegmentationTests.swift
// - Shared utilities moved to TestUtils.swift

final class LoadModel: XCTestCase {
    var model: RFModel?

    override func setUpWithError() throws {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }
    
    // Basic test to verify SDK initialization
    func testSDKInitialization() async {
        let rf = RoboflowMobile(apiKey: API_KEY)
        XCTAssertNotNil(rf, "RoboflowMobile should initialize successfully")
    }
}
