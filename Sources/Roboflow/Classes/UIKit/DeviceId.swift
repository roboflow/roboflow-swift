//
//  DeviceId.swift
//  roboflow-swift
//
//  Created by Maxwell Stone on 6/12/25.
//

import Foundation

#if os(macOS)
import IOKit
#elseif canImport(UIKit)
import UIKit
#endif

@available(macOS 12.0, *)
func getDeviceId() -> String? {
    #if os(iOS)
    return UIDevice.current.identifierForVendor?.uuidString

    #elseif os(macOS)
    let service = IOServiceGetMatchingService(kIOMainPortDefault,
                                               IOServiceMatching("IOPlatformExpertDevice"))
    guard service != 0 else { return nil }

    let key = kIOPlatformUUIDKey as CFString
    let cfStr = IORegistryEntryCreateCFProperty(service, key, kCFAllocatorDefault, 0)
    IOObjectRelease(service)

    return (cfStr?.takeUnretainedValue() as? String)

    #else
    return nil
    #endif
}
