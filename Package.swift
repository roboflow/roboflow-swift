// swift-tools-version:5.5

import PackageDescription

let package = Package(
    name: "RoboflowSwift",
    defaultLocalization: "en",
    products: [
        .library(name: "RoboflowSwift", targets: ["RoboflowSwift"]),
    ],
    targets: [
        .target(
            name: "Roboflow",
            dependencies: []),
    ]
)
