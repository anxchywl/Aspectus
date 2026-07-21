// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Aspectus",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "AspectusKit", targets: ["AspectusKit"]),
    ],
    targets: [
        // Framework-agnostic pipeline core: types, scheduling, temporal
        // filtering, confidence/fallback logic, and stage protocols.
        // Deliberately free of AVFoundation / Metal / CoreML so it builds
        // and unit-tests on any Swift toolchain (including CI without a camera).
        .target(
            name: "AspectusKit",
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
            ]
        ),
        .testTarget(
            name: "AspectusKitTests",
            dependencies: ["AspectusKit"]
        ),
    ]
)
