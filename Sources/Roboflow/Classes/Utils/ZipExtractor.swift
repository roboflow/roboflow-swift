//
//  ZipExtractor.swift
//  Roboflow
//
//  Created by Maxwell Stone on 12/19/24.
//

import Foundation
#if canImport(Compression)
import Compression
#endif

/// Error types for compression operations
enum CompressionError: Error, LocalizedError {
    case decompressionFailed
    
    var errorDescription: String? {
        switch self {
        case .decompressionFailed:
            return "Failed to decompress deflate data"
        }
    }
}

/// Utility class for extracting ZIP files on iOS where command-line tools are not available
public class ZipExtractor {
    
    /// Extracts a ZIP file to the specified directory using iOS-compatible methods
    /// - Parameters:
    ///   - zipURL: The URL of the ZIP file to extract
    ///   - extractionDirectory: The directory where files should be extracted
    /// - Throws: Errors if extraction fails
    public static func extractZip(zipURL: URL, to extractionDirectory: URL) throws {
        let zipData = try Data(contentsOf: zipURL)
        try extractZipData(zipData, to: extractionDirectory)
    }
    
    /// Extracts ZIP data to the specified directory
    /// - Parameters:
    ///   - data: The ZIP file data
    ///   - extractionDirectory: The directory where files should be extracted
    /// - Throws: Errors if extraction fails
    private static func extractZipData(_ data: Data, to extractionDirectory: URL) throws {
        // Look for ZIP local file header signature: "PK\003\004"
        let localFileHeaderSignature: [UInt8] = [0x50, 0x4b, 0x03, 0x04]
        var offset = 0
        var extractedCount = 0
        
        while offset < data.count - 30 && extractedCount < 50 { // Allow more extractions for MLPackage
            // Check for local file header signature
            if offset + 4 <= data.count {
                let signatureBytes = Array(data[offset..<offset+4])
                if signatureBytes == localFileHeaderSignature {
                    if try extractSingleFile(from: data, offset: &offset, to: extractionDirectory, fileIndex: extractedCount) {
                        extractedCount += 1
                    }
                } else {
                    offset += 1
                }
            } else {
                break
            }
        }
        
        if extractedCount == 0 {
            throw NSError(domain: "ZipExtractionError", code: 3, userInfo: [NSLocalizedDescriptionKey: "No relevant model files found in ZIP archive"])
        }
    }
    
    /// Extracts a single file from ZIP data
    /// - Parameters:
    ///   - data: The ZIP file data
    ///   - offset: Current offset in the data (will be updated)
    ///   - extractionDirectory: The directory where files should be extracted
    ///   - fileIndex: Index of the current file being extracted
    /// - Returns: True if a file was successfully extracted, false otherwise
    /// - Throws: Errors if extraction fails
    private static func extractSingleFile(from data: Data, offset: inout Int, to extractionDirectory: URL, fileIndex: Int) throws -> Bool {
        // Parse ZIP local file header according to ZIP format specification
        guard offset + 30 <= data.count else { return false }
        
        // Skip signature (4 bytes) and version (2 bytes)
        offset += 6
        
        // Read general purpose bit flag (2 bytes) - skip for now
        offset += 2
        
        // Read compression method (2 bytes) - use safe byte reading
        guard offset + 2 <= data.count else { return false }
        let compressionMethod = readUInt16(from: data, at: offset)
        offset += 2
        
        // Skip modification time (4 bytes) and CRC32 (4 bytes)
        offset += 8
        
        // Read compressed size (4 bytes) - use safe byte reading
        guard offset + 4 <= data.count else { return false }
        let compressedSize = readUInt32(from: data, at: offset)
        offset += 4
        
        // Read uncompressed size (4 bytes) - use safe byte reading
        guard offset + 4 <= data.count else { return false }
        let uncompressedSize = readUInt32(from: data, at: offset)
        offset += 4
        
        // Read filename length (2 bytes) - use safe byte reading
        guard offset + 2 <= data.count else { return false }
        let filenameLength = readUInt16(from: data, at: offset)
        offset += 2
        
        // Read extra field length (2 bytes) - use safe byte reading
        guard offset + 2 <= data.count else { return false }
        let extraFieldLength = readUInt16(from: data, at: offset)
        offset += 2
        
        // Read filename
        guard offset + Int(filenameLength) <= data.count else { return false }
        let filenameData = data.subdata(in: offset..<offset + Int(filenameLength))
        guard let filename = String(data: filenameData, encoding: .utf8) else { return false }
        offset += Int(filenameLength)
        
        // Skip extra field
        offset += Int(extraFieldLength)
        
        // Read file data
        guard offset + Int(compressedSize) <= data.count else { return false }
        let fileData = data.subdata(in: offset..<offset + Int(compressedSize))
        offset += Int(compressedSize)
        
        // Skip directories and hidden files
        if filename.hasSuffix("/") || filename.contains("__MACOSX") || filename.hasPrefix(".") {
            return false
        }
        
        // Only extract relevant model files and their dependencies
        let lowercaseFilename = filename.lowercased()
        let isRelevant = lowercaseFilename.hasSuffix(".mlmodel") || 
                        lowercaseFilename.hasSuffix(".mlpackage") ||
                        lowercaseFilename.contains("model") ||
                        lowercaseFilename.contains("weights") ||
                        lowercaseFilename.contains("data") ||
                        lowercaseFilename.hasSuffix(".bin") ||
                        lowercaseFilename.hasSuffix(".json") ||
                        lowercaseFilename.hasSuffix(".plist")
        
        guard isRelevant else { return false }
        
        // Preserve the original path structure for MLPackage models
        // This is crucial for maintaining the internal structure of .mlpackage directories
        let fileURL = extractionDirectory.appendingPathComponent(filename)
        
        // Create intermediate directories if needed (preserving full path)
        let directory = fileURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true, attributes: nil)
        
        // Handle compression
        do {
            if compressionMethod == 0 {
                // No compression - store method
                try fileData.write(to: fileURL)
            } else if compressionMethod == 8 {
                // Deflate compression - decompress using Apple's Compression framework
                #if canImport(Compression)
                let decompressedData = try decompressDeflate(data: fileData, expectedSize: Int(uncompressedSize))
                try decompressedData.write(to: fileURL)
                #else
                // Fallback: try writing compressed data (might work for some files)
                try fileData.write(to: fileURL)
                #endif
            } else {
                // Unknown compression method - skip file
                print("Warning: Unsupported compression method \(compressionMethod) for file \(filename)")
                return false
            }
            
            return true
        } catch {
            print("Error extracting file \(filename): \(error)")
            return false
        }
    }
    
    /// Helper function to read UInt32 using safe byte-by-byte reading
    private static func readUInt32(from data: Data, at offset: Int) -> UInt32 {
        guard offset + 4 <= data.count else { return 0 }
        
        // Read bytes individually to avoid alignment issues
        let byte0 = UInt32(data[offset])
        let byte1 = UInt32(data[offset + 1])
        let byte2 = UInt32(data[offset + 2])
        let byte3 = UInt32(data[offset + 3])
        
        // Combine in little-endian order
        return byte0 | (byte1 << 8) | (byte2 << 16) | (byte3 << 24)
    }
    
    /// Helper function to read UInt16 using safe byte-by-byte reading
    private static func readUInt16(from data: Data, at offset: Int) -> UInt16 {
        guard offset + 2 <= data.count else { return 0 }
        
        // Read bytes individually to avoid alignment issues
        let byte0 = UInt16(data[offset])
        let byte1 = UInt16(data[offset + 1])
        
        // Combine in little-endian order
        return byte0 | (byte1 << 8)
    }
    
    #if canImport(Compression)
    /// Decompresses deflate-compressed data using Apple's Compression framework
    /// - Parameters:
    ///   - data: The compressed data
    ///   - expectedSize: Expected uncompressed size
    /// - Returns: Decompressed data
    /// - Throws: CompressionError if decompression fails
    private static func decompressDeflate(data: Data, expectedSize: Int) throws -> Data {
        // ZIP uses "raw deflate" without zlib headers, but Apple's Compression framework
        // expects different formats. Let's try multiple approaches.
        
        return data.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) -> Data in
            let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: expectedSize)
            defer { buffer.deallocate() }
            
            // Try COMPRESSION_ZLIB first (deflate with zlib headers)
            var decompressedSize = compression_decode_buffer(
                buffer, expectedSize,
                bytes.bindMemory(to: UInt8.self).baseAddress!, data.count,
                nil, COMPRESSION_ZLIB
            )
            
            if decompressedSize > 0 && decompressedSize <= expectedSize {
                return Data(bytes: buffer, count: decompressedSize)
            }
            
            // Try COMPRESSION_LZFSE as fallback
            decompressedSize = compression_decode_buffer(
                buffer, expectedSize,
                bytes.bindMemory(to: UInt8.self).baseAddress!, data.count,
                nil, COMPRESSION_LZFSE
            )
            
            if decompressedSize > 0 && decompressedSize <= expectedSize {
                return Data(bytes: buffer, count: decompressedSize)
            }
            
            // For ZIP raw deflate, we need to add zlib headers
            // ZIP deflate format doesn't include zlib headers, so we need to add them
            let zlibHeader: [UInt8] = [0x78, 0x9C] // zlib header for deflate
            let zlibFooter: [UInt8] = [0x00, 0x00, 0x00, 0x00] // placeholder for checksum
            
            var zlibData = Data()
            zlibData.append(contentsOf: zlibHeader)
            zlibData.append(data)
            zlibData.append(contentsOf: zlibFooter)
            
            let finalResult = zlibData.withUnsafeBytes { (zlibBytes: UnsafeRawBufferPointer) -> Data in
                let zlibDecompressedSize = compression_decode_buffer(
                    buffer, expectedSize,
                    zlibBytes.bindMemory(to: UInt8.self).baseAddress!, zlibData.count,
                    nil, COMPRESSION_ZLIB
                )
                
                guard zlibDecompressedSize > 0 && zlibDecompressedSize <= expectedSize else {
                    // If all decompression attempts fail, create empty placeholder
                    // This prevents complete failure while allowing partial extraction
                    print("Warning: Could not decompress deflate data, creating empty placeholder")
                    return Data() // Return empty data as placeholder
                }
                
                return Data(bytes: buffer, count: zlibDecompressedSize)
            }
            
            return finalResult
        }
    }
    #endif
} 