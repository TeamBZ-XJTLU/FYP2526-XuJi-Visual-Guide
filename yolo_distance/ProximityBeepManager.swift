//
//  ProximityBeepManager.swift
//  yolo_distance
//
//  Created by Xu Ji on 2025/11/5.
//

import Foundation
import AVFoundation

/// 距离提示音管理：越近“滴”越快；未锁定目标时由上层调用 stop() 完全静音。
final class ProximityBeepManager {
    static let shared = ProximityBeepManager()

    // Audio
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private var outputFormat: AVAudioFormat!
    private var baseBeep: AVAudioPCMBuffer?

    // Scheduling
    private var timer: Timer?
    private var isActive = false
    private var currentInterval: TimeInterval = 0.8
    var enableNearHighPitch = true                     

    private init() {
        setupEngine()

        baseBeep = buildSineBuffer(sampleRate: outputFormat.sampleRate,
                                   channels: outputFormat.channelCount,
                                   duration: 0.08,
                                   frequency: 1000,
                                   amplitude: 0.25)
    }

    // MARK: - Public API

    /// 开始提示音
    func start() {
        guard !isActive else { return }
        isActive = true
        startEngineIfNeeded()
        scheduleNext()
    }

    /// 停止提示音并静音。
    func stop() {
        isActive = false
        timer?.invalidate()
        timer = nil
        player.stop()
    }

    /// 更新距离
    func update(distance meters: Float?) {
        guard isActive else { return }
        if let m = meters {
            currentInterval = ProximityBeepManager.interval(for: m)
        }
      
    }

    // MARK: - Engine

    private func setupEngine() {
        engine.attach(player)

        // 使用硬件输出格式
        let hw = engine.outputNode.inputFormat(forBus: 0)
        outputFormat = AVAudioFormat(standardFormatWithSampleRate: hw.sampleRate,
                                     channels: hw.channelCount)

        // 用与 outputFormat 一致的格式连线
        engine.connect(player, to: engine.mainMixerNode, format: outputFormat)
        engine.prepare()
        do { try engine.start() } catch { print("Beep engine start failed: \(error)") }
    }

    private func startEngineIfNeeded() {
        if !engine.isRunning {
            do { try engine.start() } catch { print("Beep engine restart failed: \(error)") }
        }
    }

    // MARK: - Scheduling

    private func scheduleNext() {
        guard isActive else { return }


        let minI: TimeInterval = 0.12
        let maxI: TimeInterval = 1.2
        var next: TimeInterval = currentInterval
        if next < minI { next = minI }
        if next > maxI { next = maxI }

        // 近距离阈值
        let isNear: Bool = (next <= 0.28)


        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: next, repeats: false) { [weak self] _ in
            guard let self else { return }
            self.playBeep(near: isNear)
            self.scheduleNext()
        }
        if let t = timer {
            RunLoop.main.add(t, forMode: .common)
        }
    }

    // MARK: - 播放滴声（匹配输出声道与采样率）

    private func playBeep(near: Bool) {
        if !player.isPlaying { player.play() }

        if near && enableNearHighPitch {
            if let high = buildSineBuffer(sampleRate: outputFormat.sampleRate,
                                          channels: outputFormat.channelCount,
                                          duration: 0.06,
                                          frequency: 1800,
                                          amplitude: 0.5) {
                player.scheduleBuffer(high, at: nil, options: .interrupts, completionHandler: nil)
                return
            }
        }

        if let base = baseBeep {
            player.scheduleBuffer(base, at: nil, options: .interrupts, completionHandler: nil)
        }
    }

    // MARK: - Buffer 生成（按声道数填充一致波形）

    private func buildSineBuffer(sampleRate: Double,
                                 channels: AVAudioChannelCount,
                                 duration: Double,
                                 frequency: Double,
                                 amplitude: Float) -> AVAudioPCMBuffer? {
        let framesDouble = duration * sampleRate
        let frameCount = AVAudioFrameCount(framesDouble.rounded())

        guard let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: channels),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else { return nil }
        buffer.frameLength = frameCount

        // 生成单声道波形
        let count = Int(frameCount)
        var mono = [Float](repeating: 0, count: count)
        let twoPi = 2.0 * Double.pi
        let rate = sampleRate
        for n in 0..<count {
            let t = Double(n) / rate
            mono[n] = amplitude * Float(sin(twoPi * frequency * t))
        }

        // 3ms 淡入淡出，避免爆音
        let fadeFrames = Int((0.003 * sampleRate).rounded())
        let safeFade = min(fadeFrames, count)
        for i in 0..<safeFade {
            let k = Float(i) / Float(safeFade)
            mono[i] *= k
            mono[count - 1 - i] *= k
        }

        // 复制到所有声道
        if let channelData = buffer.floatChannelData {
            for c in 0..<Int(channels) {
                let dst = channelData[c]
                mono.withUnsafeBufferPointer { src in
                    dst.assign(from: src.baseAddress!, count: count)
                }
            }
        }
        return buffer
    }

    // MARK: - 距离 → 间隔映射

    /// 0m → 0.15s（很快）；3m → 1.2s（很慢）；>3m 固定 1.2s
    private static func interval(for meters: Float) -> TimeInterval {
        let d = Double(max(0, meters))
        let clamped = min(d, 3.0)
        let t = clamped / 3.0
        let minI = 0.15
        let maxI = 1.2
        return minI + (maxI - minI) * t
    }
}
