//
//  UnsupportedOSError.swift
//

import Foundation

public struct UnsupportedOSError: Error, LocalizedError, CustomStringConvertible {

    // MARK: Stored properties
    public let message: String

    // MARK: Initialiser
    public init(
        message: String = "This feature is not available on the current operating system."
    ) {
        self.message     = message
    }

    // MARK: LocalizedError
    public var errorDescription: String? { message }

    // MARK: CustomStringConvertible
    public var description: String {
        "[UnsupportedOSError] \(message)"
    }
}

public struct DetectionError: Error, LocalizedError, CustomStringConvertible {

    // MARK: Stored properties
    public let message: String

    // MARK: Initialiser
    public init(
        message: String = "Error performing detection."
    ) {
        self.message     = message
    }

    // MARK: LocalizedError
    public var errorDescription: String? { message }

    // MARK: CustomStringConvertible
    public var description: String {
        "[DetectionError] \(message)"
    }
}

public struct ModelLoadError: Error, LocalizedError, CustomStringConvertible {

    // MARK: Stored properties
    public let message: String

    // MARK: Initialiser
    public init(
        message: String = "Error loading model."
    ) {
        self.message     = message
    }

    // MARK: LocalizedError
    public var errorDescription: String? { message }

    // MARK: CustomStringConvertible
    public var description: String {
        "[ModelLoadError] \(message)"
    }
}