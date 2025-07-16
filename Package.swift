// swift-tools-version:5.9

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
    dependencies: [],
    targets: [
        .target(
            name: "Roboflow",
            dependencies: [], path: "Sources/Roboflow"
        ),
        .testTarget(
            name: "RoboflowTests", 
            dependencies: ["Roboflow"],
            path: "Tests/RoboflowTests",
            resources: [
                .copy("assets")
            ]
        )
    ]
)
