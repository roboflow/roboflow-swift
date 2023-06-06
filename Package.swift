// swift-tools-version:5.8

import PackageDescription

let package = Package(
    name: "roboflow-swift",
    defaultLocalization: "en",

    products: [
        .library(
            name: "Roboflow",
            targets: ["Roboflow"]),
    ],
    targets: [
        .target(
            name: "Roboflow",
            path: "Sources/Roboflow"
        )
    ]
)
