//
//  ContentView.swift
//  yolo_distance
//
//  Created by Xu Ji on 2025/9/20.
//  ContentView.swift
//  yolo_distance

import SwiftUI

struct ContentView: View {

    // MARK: - 模式：待机 / 寻物
    enum AppMode {
        case standby
        case searching   
    }

    @StateObject private var voice = VoiceIntentManager()

    @State private var mode: AppMode = .standby

    // 场景描述总开关（传给 ARCameraDetectView）
    @State private var sceneNarrationEnabled = false

    // 录音开始后的 5s 超时
    @State private var listenTimeoutTimer: Timer?

    // 退出寻物时的视觉提示
    @State private var showSearchOffFlash = false

    var body: some View {
        ZStack(alignment: .top) {
            // AR 实时视图：传入 targetClass + 场景描述开关
            ARCameraDetectView(
                targetClass: voice.targetKeyword,
                enableSceneNarration: sceneNarrationEnabled
            )
            .edgesIgnoringSafeArea(.all)

            // 顶部状态条
            VStack(spacing: 8) {
                HStack(spacing: 10) {

                    // 模式显示：Standby / Searching
                    Text(mode == .standby ? "Standby" : "Searching")
                        .font(.caption.bold())
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background((mode == .standby ? Color.gray : Color.blue).opacity(0.85))
                        .foregroundColor(.white)
                        .cornerRadius(8)

                    // 录音状态
                    Label(voice.isListening ? "Listening…" : "Idle",
                          systemImage: voice.isListening ? "mic.fill" : "pause.circle")
                        .padding(.horizontal, 10).padding(.vertical, 6)
                        .background((voice.isListening ? Color.blue : Color.gray).opacity(0.85))
                        .foregroundColor(.white)
                        .cornerRadius(8)

                    
                    if !voice.targetKeyword.isEmpty {
                        Text("Target: \(voice.targetKeyword)")
                            .bold()
                            .padding(.horizontal, 10).padding(.vertical, 6)
                            .background(.red.opacity(0.85))
                            .foregroundColor(.white)
                            .cornerRadius(8)
                    }

                    Spacer()

                    // 场景描述状态（由长按切换）
                    Text(sceneNarrationEnabled ? "Scene ON" : "Scene OFF")
                        .font(.caption2.bold())
                        .padding(.horizontal, 8).padding(.vertical, 4)
                        .background((sceneNarrationEnabled ? Color.green : Color.gray).opacity(0.9))
                        .foregroundColor(.white)
                        .cornerRadius(6)
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

            // 中央“Search off” 视觉反馈（双击退出寻物时）
            if showSearchOffFlash {
                Text("Search off")
                    .font(.footnote.weight(.semibold))
                    .padding(.horizontal, 12).padding(.vertical, 8)
                    .background(.ultraThinMaterial)
                    .cornerRadius(10)
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(Color.white.opacity(0.7), lineWidth: 0.5)
                    )
                    .transition(.opacity)
                    .zIndex(10)
            }

            // 全屏手势层：
            //   - 双击：开启 / 退出寻物
            //   - 长按：切换场景描述
            Color.clear
                .contentShape(Rectangle())
                .ignoresSafeArea()
                .simultaneousGesture(
                    TapGesture(count: 2)
                        .onEnded { handleDoubleTapToggleSearch() }
                )
                .simultaneousGesture(
                    LongPressGesture(minimumDuration: 1)
                        .onEnded { _ in toggleSceneNarration() }
                )
                .accessibilityLabel("Double tap to toggle object search. Long press to toggle scene description.")
        }
        // 不再 onAppear 自动 startFlow，进入时保持待机
        .onDisappear {
            voice.stopListening()
            ProximityBeepManager.shared.stop()
            invalidateListenTimeout()
        }
        // 监听 isListening 变化，在“开始录音时”启动 5s 超时
        .onChange(of: voice.isListening) { newValue in
            handleListeningChange(newValue)
        }
    }

    // MARK: - 双击：toggle 寻物模式

    private func handleDoubleTapToggleSearch() {
        let gen = UIImpactFeedbackGenerator(style: .light)
        gen.impactOccurred()

        switch mode {
        case .standby:
            // 待机 → 进入寻物
            enterSearchingMode()

        case .searching:
            // 寻物 → 退出回待机
            exitSearchToStandby()
        }
    }

    /// 从待机进入寻物模式：播放提示语并在内部自动开始录音
    private func enterSearchingMode() {
        // 进入寻物时，关闭环境描述
        if sceneNarrationEnabled {
            sceneNarrationEnabled = false
        }

        // 停掉当前监听 + 滴声 + 超时
        voice.stopListening()
        ProximityBeepManager.shared.stop()
        invalidateListenTimeout()

        voice.recognizedText = ""
        voice.targetKeyword = ""
        voice.errorMessage = nil

        mode = .searching
        voice.startFlow()   // 内部：TTS "What are you looking for?" → 自动 startOneShotListening()
    }

    /// 从寻物模式主动退出回待机
    private func exitSearchToStandby() {
        mode = .standby
        invalidateListenTimeout()
        voice.stopListening()
        ProximityBeepManager.shared.stop()
        voice.targetKeyword = ""
        voice.errorMessage = nil

        withAnimation(.easeInOut(duration: 0.15)) { showSearchOffFlash = true }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
            withAnimation(.easeOut(duration: 0.2)) { showSearchOffFlash = false }
        }
    }

    // MARK: - 长按：切换场景描述开关

    private func toggleSceneNarration() {
        let gen = UIImpactFeedbackGenerator(style: .medium)
        gen.impactOccurred()
        sceneNarrationEnabled.toggle()
    }

    // MARK: - 监听 isListening：在“开始录音”时启动 5s 超时

    private func handleListeningChange(_ listening: Bool) {
        guard mode == .searching else {
            invalidateListenTimeout()
            return
        }

        if listening {

            scheduleListenTimeout()
        } else {

            invalidateListenTimeout()
        }
    }

    /// 录音开始后 5s 内如果 recognizedText 仍然是空，就退回待机
    private func scheduleListenTimeout() {
        invalidateListenTimeout()

        listenTimeoutTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: false) { _ in
            let hasSpeech = !voice.recognizedText
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .isEmpty

            if mode == .searching && !hasSpeech {
                // 视为“没说话” → 回到待机模式
                mode = .standby
                voice.stopListening()
                ProximityBeepManager.shared.stop()
                voice.targetKeyword = ""
                voice.errorMessage = "No speech detected, back to standby."
            }
        }
        if let t = listenTimeoutTimer {
            RunLoop.main.add(t, forMode: .common)
        }
    }

    private func invalidateListenTimeout() {
        listenTimeoutTimer?.invalidate()
        listenTimeoutTimer = nil
    }
}
