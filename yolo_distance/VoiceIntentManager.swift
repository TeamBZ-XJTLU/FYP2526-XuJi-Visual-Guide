//
//  VoiceIntentManager.swift
//  yolo_distance
//
//  Created by Xu Ji on 2025/11/5.
//
import Foundation
import SwiftUI
import AVFoundation
import Speech

@MainActor
final class VoiceIntentManager: NSObject, ObservableObject, AVSpeechSynthesizerDelegate {

    // MARK: - Published
    @Published var isListening: Bool = false
    @Published var recognizedText: String = ""
    @Published var targetKeyword: String = ""
    @Published var errorMessage: String? = nil

    // MARK: - TTS / STT
    private let tts = AVSpeechSynthesizer()
    private var speechRecognizer: SFSpeechRecognizer? = SFSpeechRecognizer(locale: Locale(identifier: "en_US"))


    private enum TTSPhase { case prompt, confirm }
    private var ttsPhase: TTSPhase? = nil

    // MARK: - Audio Engine graph
    private let audioEngine = AVAudioEngine()
    private let mixer = AVAudioMixerNode()
    private var mixerAttached = false

    // MARK: - Recognition
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var debounceTimer: Timer?

    private let promptText = "What are you looking for?"
    private var finishedOnce = false   // 保证本轮只完成一次

    override init() {
        super.init()
        tts.delegate = self
    }

    // MARK: - Public API
    func startFlow() {
        Task { @MainActor in
            print("[Voice] startFlow()")
            do {
                try await requestPermissions()
                recognizedText = ""
                targetKeyword = ""
                errorMessage = nil
                finishedOnce = false
                speakPrompt()  // 仅这一次会在播报结束后开始监听
            } catch {
                let msg = "Permission denied: \(error.localizedDescription)"
                errorMessage = msg
                print("[Voice][ERR] \(msg)")
            }
        }
    }

    func stopListening() {
        internalStopAudioGraph()
        configureAudioSessionForPlayback()
    }

    // MARK: - Permissions
    private func requestPermissions() async throws {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            AVAudioSession.sharedInstance().requestRecordPermission { granted in
                granted ? cont.resume() : cont.resume(throwing: NSError(domain: "MicDenied", code: 1))
            }
        }
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            SFSpeechRecognizer.requestAuthorization { status in
                status == .authorized ? cont.resume() : cont.resume(throwing: NSError(domain: "SpeechDenied", code: 2))
            }
        }
    }

    // MARK: - TTS
    private func speakPrompt() {
        ttsPhase = .prompt
        configureAudioSessionForPlayback()
        if tts.isSpeaking { tts.stopSpeaking(at: .immediate) }
        let u = AVSpeechUtterance(string: promptText)
        u.voice = AVSpeechSynthesisVoice(language: "en-US")
        u.rate = AVSpeechUtteranceDefaultSpeechRate
        print("[Voice] TTS(prompt) → \(promptText)")
        tts.speak(u)
    }

    // 播报“Looking for …”，播完不再监听
    private func speakLooking(for target: String) {
        guard !target.isEmpty else { return }
        ttsPhase = .confirm
        safelySetAudioSession(category: .playback,
                              mode: .spokenAudio,
                              options: [.duckOthers]) // 不打断滴声，仅轻度压低
        if tts.isSpeaking { tts.stopSpeaking(at: .immediate) }
        let phrase = "Looking for \(target)"
        let utt = AVSpeechUtterance(string: phrase)
        utt.voice = AVSpeechSynthesisVoice(language: "en-US")
        utt.rate = AVSpeechUtteranceDefaultSpeechRate
        print("[Voice] TTS(confirm) → \(phrase)")
        tts.speak(utt)
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        let phase = ttsPhase
        ttsPhase = nil
        if phase == .prompt {
            print("[Voice] TTS(prompt) finished → start one-shot listening")
            startOneShotListening()    // 仅此一次会开启监听
        } else {
            print("[Voice] TTS(confirm/none) finished → do nothing")
        }
    }

    // MARK: - AudioSession helpers
    private func safelySetAudioSession(category: AVAudioSession.Category,
                                       mode: AVAudioSession.Mode,
                                       options: AVAudioSession.CategoryOptions = []) {
        let session = AVAudioSession.sharedInstance()
        do { try session.setActive(false, options: .notifyOthersOnDeactivation) } catch { }
        do {
            try session.setCategory(category, mode: mode, options: options)
            try session.setPreferredSampleRate(44100)
            try session.setPreferredInputNumberOfChannels(1)
            try session.setPreferredIOBufferDuration(0.02)
            try session.setActive(true)
        } catch {
            print("[Voice][ERR] AudioSession set failed: \(error)")
        }
    }

    private func configureAudioSessionForPlayback() {
        safelySetAudioSession(category: .playback, mode: .spokenAudio, options: [.duckOthers])
    }

    private func configureAudioSessionForRecord() {
        safelySetAudioSession(category: .playAndRecord,
                              mode: .measurement,
                              options: [.defaultToSpeaker, .allowBluetooth, .duckOthers])
    }

    // MARK: - One-shot listening (ONLY once after prompt)
    private func startOneShotListening() {
        guard let recognizer = speechRecognizer, recognizer.isAvailable else {
            let msg = "Speech recognizer not available."
            errorMessage = msg
            print("[Voice][ERR] \(msg)")
            return
        }
        if isListening { return }

        if tts.isSpeaking { tts.stopSpeaking(at: .immediate) }

        internalStopAudioGraph()
        configureAudioSessionForRecord()

        audioEngine.stop()
        audioEngine.reset()

        if !mixerAttached || mixer.engine == nil {
            audioEngine.attach(mixer)
            mixerAttached = true
        }

        let inputNode = audioEngine.inputNode
        audioEngine.disconnectNodeOutput(inputNode)
        audioEngine.disconnectNodeOutput(mixer)
        audioEngine.connect(inputNode, to: mixer, format: nil)
        audioEngine.connect(mixer, to: audioEngine.mainMixerNode, format: nil)

        let tapFormat = mixer.outputFormat(forBus: 0)
        guard tapFormat.sampleRate > 0, tapFormat.channelCount > 0 else {
            let msg = "Invalid tap format: rate=\(tapFormat.sampleRate), ch=\(tapFormat.channelCount)"
            errorMessage = msg
            print("[Voice][ERR] \(msg)")
            return
        }

        let req = SFSpeechAudioBufferRecognitionRequest()
        req.shouldReportPartialResults = true
        self.request = req

        if mixer.engine != nil {
            mixer.removeTap(onBus: 0)
            mixer.installTap(onBus: 0, bufferSize: 1024, format: tapFormat) { [weak self] buffer, _ in
                self?.request?.append(buffer)
            }
        } else {
            let msg = "Mixer not attached to engine."
            errorMessage = msg
            print("[Voice][ERR] \(msg)")
            return
        }

        do {
            audioEngine.prepare()
            try audioEngine.start()
            isListening = true
            print("[Voice] Listening started (one-shot)")
        } catch {
            let msg = "Audio engine start failed: \(error)"
            errorMessage = msg
            print("[Voice][ERR] \(msg)")
            return
        }

        self.task = recognizer.recognitionTask(with: req) { [weak self] result, err in
            guard let self else { return }

            if let err = err {
                if !self.finishedOnce {
                    self.errorMessage = "Recognition error: \(err.localizedDescription)"
                    print("[Voice][ERR] \(err.localizedDescription)")
                }
                self.stopListening()
                return
            }

            if let result = result {
                let text = result.bestTranscription.formattedString
                self.recognizedText = text
                print("[Voice] partial/final → \"\(text)\"  final=\(result.isFinal)")

                // 兜底防抖：说完 0.7s 自动收尾
                self.debounceTimer?.invalidate()
                self.debounceTimer = Timer.scheduledTimer(withTimeInterval: 0.7, repeats: false) { [weak self] _ in
                    self?.finishOnce(with: text)
                }
                RunLoop.main.add(self.debounceTimer!, forMode: .common)

                if result.isFinal {
                    self.finishOnce(with: text)
                }
            }
        }
    }

    // MARK: - Finish once (stop mic forever in this round)
    private func finishOnce(with text: String) {
        guard !finishedOnce else { return }
        finishedOnce = true

        stopListening() // 立刻停麦，并且本轮不会再次开启

        let keyword = extractKeyword(from: text)
        self.targetKeyword = keyword
        print("[Voice] targetKeyword = \"\(keyword)\"")

        // 播报确认，不会引发监听
        speakLooking(for: keyword)
    }

    // MARK: - Text Utils
    private func extractKeyword(from text: String) -> String {
        // 取最后一个“实词”
        let stop = Set(["a","an","the","please","thanks","thank","you"])
        let tokens = text
            .lowercased()
            .replacingOccurrences(of: "[^a-z0-9\\s-]", with: " ", options: .regularExpression)
            .split(separator: " ")
            .map(String.init)
            .filter { !$0.isEmpty && !stop.contains($0) }

        return tokens.last ?? text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Stop & Cleanup
    private func internalStopAudioGraph() {
        debounceTimer?.invalidate()
        debounceTimer = nil

        request?.endAudio()
        task?.cancel()
        request = nil
        task = nil

        if mixer.engine != nil {
            mixer.removeTap(onBus: 0)
        }

        if audioEngine.isRunning {
            audioEngine.stop()
        }
        audioEngine.reset()

        if isListening {
            print("[Voice]  Listening stopped")
        }
        isListening = false
    }
}
