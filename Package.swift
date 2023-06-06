// swift-tools-version:5.8

import PackageDescription

let package = Package(
    name: "Roboflow",
    defaultLocalization: "en",
    
    products: [
        .library(
            name: "RoboflowSwift",
            targets: ["RoboflowSwift"]),
    ],
    dependencies: [
    ],
    targets: [
        .target(
            name: "RoboflowSwift",
            path: "Sources/Roboflow"
        )
    ]
)
