// swift-tools-version:5.8

import PackageDescription

let package = Package(
    name: "roboflow-swift",
    defaultLocalization: "en",
    platforms: [
        .iOS(.v16),
    ],
    
    products: [
        .library(
            name: "Roboflow",
            targets: ["Roboflow"]),
    ],
    dependencies: [
        .package(url: "git@github.com:qoncept/TensorSwift.git", from: "0.2.0"),
    ],
    targets: [
        .target(
            name: "Roboflow",
            path: "Sources/Roboflow"
        ),
        .testTarget(name:"RoboflowTests", dependencies: ["Roboflow"])
    ]
)
