//
//  ContentView.swift
//  yolo_distance
//
//  Created by Xu Ji on 2025/9/20.
//

import SwiftUI

struct ContentView: View {
    @StateObject private var voice = VoiceIntentManager()
    @State private var showRestartFlash = false

    var body: some View {
        ZStack(alignment: .top) {
            // AR 实时视图
            ARCameraDetectView(targetClass: voice.targetKeyword)
                .edgesIgnoringSafeArea(.all)

            // 顶部状态条
            VStack(spacing: 8) {
                HStack(spacing: 10) {
                    Label(voice.isListening ? "Listening…" : "Idle",
                          systemImage: voice.isListening ? "mic.fill" : "pause.circle")
                        .padding(.horizontal, 10).padding(.vertical, 6)
                        .background((voice.isListening ? Color.blue : Color.gray).opacity(0.85))
                        .foregroundColor(.white)
                        .cornerRadius(8)

                    if !voice.recognizedText.isEmpty {
                        Text("Heard: \(voice.recognizedText)")
                            .lineLimit(1)
                            .padding(.horizontal, 10).padding(.vertical, 6)
                            .background(.black.opacity(0.6))
                            .foregroundColor(.white)
                            .cornerRadius(8)
                    }

                    if !voice.targetKeyword.isEmpty {
                        Text("Target: \(voice.targetKeyword)")
                            .bold()
                            .padding(.horizontal, 10).padding(.vertical, 6)
                            .background(.red.opacity(0.85))
                            .foregroundColor(.white)
                            .cornerRadius(8)
                    }

                    Spacer()
                }

                if let msg = voice.errorMessage {
                    Text(msg).font(.footnote)
                        .padding(6)
                        .background(.yellow.opacity(0.9))
                        .foregroundColor(.black)
                        .cornerRadius(6)
                }
            }
            .padding(.top, 50)
            .padding(.horizontal, 12)

            // 中央“Restarted” 视觉反馈
            if showRestartFlash {
                Text("Restarted")
                    .font(.footnote.weight(.semibold))
                    .padding(.horizontal, 12).padding(.vertical, 8)
                    .background(.ultraThinMaterial)
                    .cornerRadius(10)
                    .overlay(
                        RoundedRectangle(cornerRadius: 10).stroke(Color.white.opacity(0.7), lineWidth: 0.5)
                    )
                    .transition(.opacity)
                    .zIndex(10)
            }

            // 全屏双击手势层（不阻挡其它控件的点按）
            Color.clear
                .contentShape(Rectangle())
                .ignoresSafeArea()
                .onTapGesture(count: 2) { restartFlow() }
                .accessibilityLabel("Double tap to restart voice query")
        }
        .onAppear { voice.startFlow() }
        .onDisappear {
            voice.stopListening()
            ProximityBeepManager.shared.stop()
        }
    }

    // MARK: - Restart logic
    private func restartFlow() {
        // 触觉反馈
        let gen = UIImpactFeedbackGenerator(style: .light)
        gen.impactOccurred()

        // 停掉当前监听与滴声
        voice.stopListening()
        ProximityBeepManager.shared.stop()

        // 清空状态
        voice.recognizedText = ""
        voice.targetKeyword = ""
        voice.errorMessage = nil

        // 视觉提示
        withAnimation(.easeInOut(duration: 0.15)) { showRestartFlash = true }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
            withAnimation(.easeOut(duration: 0.2)) { showRestartFlash = false }
        }

        // 重新开始一轮：播报提示 → 单次识别
        voice.startFlow()
    }
}
