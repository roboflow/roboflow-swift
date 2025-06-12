//
//  UIImage+Extension.swift
//  Roboflow
//
//  Created by Nicholas Arner on 7/11/22.
//

#if canImport(UIKit)
import UIKit

import VideoToolbox

extension UIImage {
    //Create a UIImage out of a CVPixelBuffer
    public convenience init?(pixelBuffer: CVPixelBuffer) {
        var cgImage: CGImage?
        VTCreateCGImageFromCVPixelBuffer(pixelBuffer, options: nil, imageOut: &cgImage)

        guard let cgImage = cgImage else {
            return nil
        }

            self.init(cgImage: cgImage)
    }
}
#endif
