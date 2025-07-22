//
//  ContentView.swift
//  Roboflow Swift Benchmark
//
//  Created by Maxwell Stone on 7/21/25.
//

import SwiftUI
import Roboflow

struct ContentView: View {
    @State private var statusMessage = "Starting..."
    @State private var isLoading = false
    @State private var model: RFDetrObjectDetectionModel?
    @State private var inferenceCount = 0
    @State private var inferenceRunning = false
    @State private var lastInferenceTime: Double = 0.0
    
    // API Key and model configuration from TestUtils
    private let apiKey = "rf_EsVTlbAbaZPLmAFuQwWoJgFpMU82"
    private let modelName = "hard-hat-sample-txcpu"
    private let modelVersion = 7
    
    var body: some View {
        VStack(spacing: 20) {
            Text("RFDETR Benchmark")
                .font(.title)
                .fontWeight(.bold)
            
            Text(statusMessage)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding()
            
            if isLoading {
                ProgressView("Loading model...")
                    .progressViewStyle(CircularProgressViewStyle())
            }
            
            if let _ = model, !isLoading {
                VStack(spacing: 15) {
                    Text("Model loaded successfully!")
                        .foregroundColor(.green)
                        .fontWeight(.semibold)
                    
                    Button(action: runRandomInference) {
                        HStack {
                            if inferenceRunning {
                                ProgressView()
                                    .scaleEffect(0.8)
                                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            }
                            Text(inferenceRunning ? "Running..." : "Run Random Inference")
                        }
                        .foregroundColor(.white)
                        .padding()
                        .background(inferenceRunning ? Color.gray : Color.blue)
                        .cornerRadius(10)
                    }
                    .disabled(inferenceRunning)
                    
                    if inferenceCount > 0 {
                        VStack(spacing: 5) {
                            Text("Inference Count: \(inferenceCount)")
                                .fontWeight(.medium)
                            
                            if lastInferenceTime > 0 {
                                Text("Last Inference: \(String(format: "%.3f", lastInferenceTime))s")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    
                    Button(action: runContinuousInference) {
                        Text("Run Continuous Inference")
                            .foregroundColor(.white)
                            .padding()
                            .background(Color.orange)
                            .cornerRadius(10)
                    }
                    .disabled(inferenceRunning)
                }
            }
        }
        .padding()
        .onAppear {
            loadModel()
        }
    }
    
    private func loadModel() {
        guard !isLoading else { return }
        
        isLoading = true
        statusMessage = "Loading RFDETR model..."
        
        Task {
            do {
                let rf = RoboflowMobile(apiKey: apiKey)
                let (loadedModel, error, _, _) = await rf.load(model: modelName, modelVersion: modelVersion)
                
                await MainActor.run {
                    if let error = error {
                        statusMessage = "Failed to load model: \(error.localizedDescription)"
                        isLoading = false
                        return
                    }
                    
                    guard let rfDetrModel = loadedModel as? RFDetrObjectDetectionModel else {
                        statusMessage = "Model is not an RFDETR model"
                        isLoading = false
                        return
                    }
                    
                    // Configure the model
                    rfDetrModel.configure(threshold: 0.5, overlap: 0.5, maxObjects: 20)
                    
                    self.model = rfDetrModel
                    statusMessage = "Model loaded and ready for inference"
                    isLoading = false
                }
            } catch {
                await MainActor.run {
                    statusMessage = "Error loading model: \(error.localizedDescription)"
                    isLoading = false
                }
            }
        }
    }
    
    private func runRandomInference() {
        guard let model = model, !inferenceRunning else { return }
        
        inferenceRunning = true
        statusMessage = "Running inference on random noise..."
        
        let startTime = Date()
        
        model.detectWithRandomNoise { predictions, error in
            DispatchQueue.main.async {
                let endTime = Date()
                let inferenceTime = endTime.timeIntervalSince(startTime)
                
                self.lastInferenceTime = inferenceTime
                self.inferenceRunning = false
                self.inferenceCount += 1
                
                if let error = error {
                    self.statusMessage = "Inference failed: \(error.localizedDescription)"
                } else {
                    let detectionCount = predictions?.count ?? 0
                    self.statusMessage = "Inference complete!\nDetected \(detectionCount) objects\nTime: \(String(format: "%.3f", inferenceTime))s"
                }
            }
        }
    }
    
    private func runContinuousInference() {
        guard let model = model else { return }
        
        Task {
            await MainActor.run {
                statusMessage = "Running continuous inference..."
                inferenceRunning = true
            }
            
            // Run 10 continuous inferences
            for i in 1...10 {
                let startTime = Date()
                
                await withCheckedContinuation { continuation in
                    model.detectWithRandomNoise { predictions, error in
                        continuation.resume()
                    }
                }
                
                let endTime = Date()
                let inferenceTime = endTime.timeIntervalSince(startTime)
                
                await MainActor.run {
                    self.inferenceCount += 1
                    self.lastInferenceTime = inferenceTime
                    self.statusMessage = "Continuous inference \(i)/10\nLast time: \(String(format: "%.3f", inferenceTime))s"
                }
                
                // Small delay between inferences
                try? await Task.sleep(nanoseconds: 100_000_000) // 0.1 seconds
            }
            
            await MainActor.run {
                self.inferenceRunning = false
                self.statusMessage = "Completed 10 continuous inferences!\nTotal inferences: \(self.inferenceCount)"
            }
        }
    }
}

#Preview {
    ContentView()
}
