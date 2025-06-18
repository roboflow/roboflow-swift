//
//  CVPixelBuffer+Size.swift
//  roboflow-swift
//
//  Created by Maxwell Stone on 6/12/25.
//
import Foundation
import Vision

extension CVPixelBuffer {
    func height() -> Int {
        CVPixelBufferGetHeight(self)
    }
    
    func width() -> Int {
        CVPixelBufferGetWidth(self)
    }
}
