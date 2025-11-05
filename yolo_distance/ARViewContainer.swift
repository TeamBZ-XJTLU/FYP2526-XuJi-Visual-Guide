import SwiftUI
import ARKit
import Vision
import CoreML
import simd

// MARK: - SwiftUI wrapper of ARSCNView
struct ARCameraDetectView: UIViewRepresentable {
    var targetClass: String = ""
    func makeCoordinator() -> Coordinator {
        Coordinator(targetClass: targetClass)  
    }

    func makeUIView(context: Context) -> ARSCNView {
        let arView = ARSCNView()
        arView.scene = SCNScene()
        arView.automaticallyUpdatesLighting = true

        // overlay 用于 2D 绘制
        context.coordinator.overlay.frame = UIScreen.main.bounds
        arView.layer.addSublayer(context.coordinator.overlay)

        // 绑定 session 委托
        arView.session.delegate = context.coordinator
        context.coordinator.arView = arView

        // 配置 AR 会话
        let config = ARWorldTrackingConfiguration()
        config.planeDetection = [.horizontal, .vertical]
        config.environmentTexturing = .automatic
        config.worldAlignment = .gravity
        if ARWorldTrackingConfiguration.supportsFrameSemantics([.sceneDepth, .smoothedSceneDepth]) {
            config.frameSemantics.insert(.sceneDepth)
            config.frameSemantics.insert(.smoothedSceneDepth)
        }

        arView.session.run(config, options: [.resetTracking, .removeExistingAnchors])
        arView.debugOptions = []
        context.coordinator.loadModel()
        return arView
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {
        context.coordinator.targetClass = targetClass
    }

    // MARK: - Coordinator
    final class Coordinator: NSObject, ARSessionDelegate {


        var targetClass: String

        init(targetClass: String) {
            self.targetClass = targetClass
        }

        private let drawEveryNFrames = 2
        private let medianWindow = 7
        private let confidenceThreshold: UInt8 = 2

        private let sampleGridHalf = 2
        private let sampleStepPx: CGFloat = 2
        private let outlierAbsThreshM: Float = 0.25
        private let outlierRelThresh: Float = 0.25
        private let minInliersForAvg = 5

        private let planeHeightThreshM: Float = 0.10
        private let planeNearFallbackM: Float = 0.18
        private let drawSupportDot = true
        private let drawPlaneOutline = true

        private let beeper = ProximityBeepManager.shared

        weak var arView: ARSCNView?
        let overlay = CALayer()

        private var vnModel: VNCoreMLModel!
        private var request: VNCoreMLRequest!
        private var inflight = false
        private var frameCount = 0
        private var planeMap: [UUID: ARPlaneAnchor] = [:]

        struct Detection {
            let rect: CGRect
            let label: String
            let score: Float
            let centerPx: CGPoint
            let centerWorld: simd_float3?
            let planeClass: String?
            let planeHorizontalDistM: Float?
            let camDistM: Float?
            let planeOutlinePx: [CGPoint]?
            let planeEdgePx: (CGPoint, CGPoint)?
            let supportWorld: simd_float3?
            let edgeWorld: (simd_float3, simd_float3)?
            let supportToEdgeDistM: Float?
            let planeAnchor: ARPlaneAnchor?
        }

        // MARK: - Load YOLO Model
        func loadModel() {
            do {
                let coreMLModel = try yolov8s_worldv2(configuration: MLModelConfiguration()).model
                vnModel = try VNCoreMLModel(for: coreMLModel)
                request = VNCoreMLRequest(model: vnModel) { [weak self] req, _ in
                    self?.handleDetections(req: req)
                }
                request.imageCropAndScaleOption = .scaleFill
            } catch {
                fatalError("加载 CoreML 模型失败：\(error)")
            }
        }

        // MARK: - AR Session Updates
        func session(_ session: ARSession, didUpdate frame: ARFrame) {
            frameCount += 1
            if inflight || frameCount % drawEveryNFrames != 0 { return }
            inflight = true

            let io = currentInterfaceOrientation(of: arView!)
            let cgOri = cgImageOrientation(for: io)
            let handler = VNImageRequestHandler(cvPixelBuffer: frame.capturedImage, orientation: cgOri)

            guard let safeRequest = self.request else { inflight = false; return }

            DispatchQueue.main.async {
                do {
                    try handler.perform([safeRequest])
                } catch {
                    print("Vision failed: \(error)")
                }
                self.inflight = false
            }


        }

        func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
            for a in anchors {
                if let p = a as? ARPlaneAnchor { planeMap[p.identifier] = p }
            }
        }
        func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
            for a in anchors {
                if let p = a as? ARPlaneAnchor { planeMap[p.identifier] = p }
            }
        }
        func session(_ session: ARSession, didRemove anchors: [ARAnchor]) {
            for a in anchors {
                if let p = a as? ARPlaneAnchor { planeMap.removeValue(forKey: p.identifier) }
            }
        }

        // MARK: - Detection Handler
        private func handleDetections(req: VNRequest) {
            guard let arView, let frame = arView.session.currentFrame else { inflight = false; return }
            guard let results = req.results else { inflight = false; return }

            let objs = results.compactMap { $0 as? VNRecognizedObjectObservation }
            let viewSize = arView.bounds.size
            let depthPB = frame.sceneDepth?.depthMap
            let confPB  = frame.sceneDepth?.confidenceMap

            var dets: [Detection] = []

            for ob in objs {
                let bbox = ob.boundingBox
                let label = ob.labels.first?.identifier ?? "object"
                let score = ob.labels.first?.confidence ?? 0
                let rectPx = vnRectToPixel(bbox: bbox, in: viewSize)
                let centerPx = CGPoint(x: rectPx.midX, y: rectPx.midY)
                let supportPx = CGPoint(x: rectPx.midX, y: rectPx.maxY - 2)

                var worldMedianCenter: simd_float3? = nil
                var worldSupport: simd_float3? = nil
                if let depthPB {
                    if let res = robustWorldAndDistanceAround(center: centerPx,
                                                              frame: frame,
                                                              viewSize: viewSize,
                                                              depthPB: depthPB,
                                                              confPB: confPB) {
                        worldMedianCenter = res.world
                    }
                    if let res = robustWorldAndDistanceAround(center: supportPx,
                                                              frame: frame,
                                                              viewSize: viewSize,
                                                              depthPB: depthPB,
                                                              confPB: confPB) {
                        worldSupport = res.world
                    }
                }

                var planeMark: String? = nil
                if let ws = worldSupport { planeMark = markHorizontalPlane(pointWorld: ws) }
                if planeMark == nil { planeMark = markPlaneViaRaycast(screenPt: supportPx) }

                var planeHorizDist: Float? = nil
                var camDist: Float? = nil
                var planeEdgePx: (CGPoint, CGPoint)? = nil
                var planeOutlinePx: [CGPoint]? = nil
                var edgeWorld: (simd_float3, simd_float3)? = nil
                var selectedAnchor: ARPlaneAnchor? = nil

                if let wc = worldMedianCenter {
                    let camT = frame.camera.transform
                    let camWorld = simd_float3(camT.columns.3.x, camT.columns.3.y, camT.columns.3.z)
                    camDist = simd_length(wc - camWorld)
                }

             


                if let ws = worldSupport,
                   let anchor = pickHorizontalAnchorForSupport(worldSupport: ws, supportPx: supportPx) {
                    selectedAnchor = anchor
                    let (edge2, poly2, edgeW) = nearestEdgeAndOutlineScreen(anchor: anchor, supportWorld: ws)
                    planeEdgePx = edge2
                    planeOutlinePx = poly2
                    edgeWorld = edgeW
                }
                if let edgeW = edgeWorld {
                    let camT = frame.camera.transform
                    let camWorld = simd_float3(camT.columns.3.x, camT.columns.3.y, camT.columns.3.z)

                    let a = simd_float2(edgeW.0.x, edgeW.0.z)
                    let b = simd_float2(edgeW.1.x, edgeW.1.z)
                    let cam2D = simd_float2(camWorld.x, camWorld.z)

                    let dist = pointToSegmentDistance2D(cam2D, a, b)
                    planeHorizDist = dist
                }
                dets.append(Detection(rect: rectPx,
                                      label: label,
                                      score: score,
                                      centerPx: centerPx,
                                      centerWorld: worldMedianCenter,
                                      planeClass: planeMark,
                                      planeHorizontalDistM: planeHorizDist,
                                      camDistM: camDist,
                                      planeOutlinePx: planeOutlinePx,
                                      planeEdgePx: planeEdgePx,
                                      supportWorld: worldSupport,
                                      edgeWorld: edgeWorld,
                                      supportToEdgeDistM: nil,
                                      planeAnchor: selectedAnchor))
            }

            DispatchQueue.main.async {
                self.drawOverlay(dets: dets, viewSize: viewSize)
                self.inflight = false
            }
            // —— 选出与 targetClass 匹配的目标最近距离（相机距离）
            var nearestTargetDist: Float? = nil
            if !targetClass.isEmpty {
                let matches = dets.filter { $0.label.lowercased().contains(self.targetClass.lowercased()) }
                if let best = matches.compactMap({ $0.camDistM }).min() {
                    nearestTargetDist = best
                }
            }

            // 根据是否识别到目标来控制 Beep
            if let dist = nearestTargetDist {
                // 首次激活
                beeper.start()
                beeper.update(distance: dist)
            } else {
                beeper.stop()
            }

        }

        // MARK: - 绘制叠加层
        private func drawOverlay(dets: [Detection], viewSize: CGSize) {
            overlay.frame = CGRect(origin: .zero, size: viewSize)
            overlay.sublayers?.forEach { $0.removeFromSuperlayer() }

            let limitedDets = dets.prefix(2)

            // 平面轮廓与橙色边
            for d in limitedDets {
                if drawPlaneOutline, let poly = d.planeOutlinePx, poly.count >= 3 {
                    let path = UIBezierPath()
                    path.move(to: poly[0])
                    for i in 1..<poly.count { path.addLine(to: poly[i]) }
                    path.close()
                    let outline = CAShapeLayer()
                    outline.path = path.cgPath
                    outline.lineWidth = 1.5
                    outline.strokeColor = UIColor.systemTeal.withAlphaComponent(0.9).cgColor
                    outline.fillColor = UIColor.clear.cgColor
                    overlay.addSublayer(outline)
                }
                if let edge = d.planeEdgePx {
                    let p = UIBezierPath()
                    p.move(to: edge.0)
                    p.addLine(to: edge.1)
                    let hl = CAShapeLayer()
                    hl.path = p.cgPath
                    hl.lineWidth = 4.0
                    hl.strokeColor = UIColor.systemOrange.withAlphaComponent(0.95).cgColor
                    overlay.addSublayer(hl)
                }
            }

            // 框与标签
            for d in limitedDets {
                let isTarget = !targetClass.isEmpty && d.label.lowercased().contains(targetClass.lowercased())

                // 外框
                let box = CAShapeLayer()
                box.path = UIBezierPath(rect: d.rect).cgPath
                if isTarget {
                    box.lineWidth = 4
                    box.strokeColor = UIColor.systemRed.cgColor
                } else {
                    box.lineWidth = 2
                    box.strokeColor = UIColor.systemGreen.cgColor
                }
                box.fillColor = UIColor.clear.cgColor
                overlay.addSublayer(box)

                // 高亮目标的顶部标识
                if isTarget {
                    let star = CATextLayer()
                    star.string = "⭐️ \(d.label)"
                    star.fontSize = 14
                    star.foregroundColor = UIColor.white.cgColor
                    star.backgroundColor = UIColor.systemRed.withAlphaComponent(0.8).cgColor
                    star.cornerRadius = 4
                    star.alignmentMode = .center
                    star.contentsScale = UIScreen.main.scale
                    star.frame = CGRect(x: d.rect.minX,
                                        y: max(d.rect.minY - 24, 0),
                                        width: 100, height: 20)
                    overlay.addSublayer(star)
                }

                // 支撑点
                if drawSupportDot {
                    let dot = CAShapeLayer()
                    let r: CGFloat = 3
                    let sp = CGPoint(x: d.rect.midX, y: d.rect.maxY - 2)
                    dot.path = UIBezierPath(ovalIn: CGRect(x: sp.x - r, y: sp.y - r, width: r*2, height: r*2)).cgPath
                    dot.fillColor = UIColor.cyan.cgColor
                    overlay.addSublayer(dot)
                }

                // 平面信息标签
                let planeText: String = {
                    switch d.planeClass {
                    case "ground": return "floor"
                    case "plane":  return "plane checked"
                    default:       return "off-plane"
                    }
                }()

                var lines: [String] = []
                lines.append("\(d.label)  \(Int(d.score * 100))%")
                lines.append(planeText)
                if let m = d.planeHorizontalDistM, d.planeClass == "plane" {
                    lines.append(String(format: "planeDist: %.02fm", m))
                }
                if let m = d.camDistM {
                    lines.append(String(format: "camDist: %.02fm", m))
                }

                let text = lines.joined(separator: "\n")
                let tag = CATextLayer()
                tag.contentsScale = UIScreen.main.scale
                tag.fontSize = 14
                tag.alignmentMode = .left
                tag.isWrapped = true
                tag.truncationMode = .end
                tag.foregroundColor = UIColor.white.cgColor
                tag.backgroundColor = UIColor.black.withAlphaComponent(0.65).cgColor
                tag.cornerRadius = 4

                let maxWidth = max(140.0, d.rect.width)
                let attr = [NSAttributedString.Key.font: UIFont.systemFont(ofSize: 14)]
                let bound = (text as NSString).boundingRect(
                    with: CGSize(width: maxWidth, height: .greatestFiniteMagnitude),
                    options: [.usesLineFragmentOrigin, .usesFontLeading],
                    attributes: attr, context: nil
                )
                tag.string = text
                tag.frame = CGRect(x: d.rect.minX,
                                   y: max(d.rect.minY - bound.height - 6, 0),
                                   width: maxWidth + 10,
                                   height: ceil(bound.height) + 6)
                overlay.addSublayer(tag)
            }

            // 距离连线
            drawAllPairLinks(dets: Array(limitedDets))
        }


        private func drawAllPairLinks(dets: [Detection]) {
            for i in 0..<dets.count {
                for j in (i+1)..<dets.count {
                    guard let pa = dets[i].centerWorld, let pb = dets[j].centerWorld else { continue }
                    let d3 = simd_length(pa - pb)
                    drawLineWithLabel(a: dets[i], b: dets[j], distance: d3)
                }
            }
        }

        private func drawLineWithLabel(a: Detection, b: Detection, distance: Float) {
            let path = UIBezierPath()
            path.move(to: a.centerPx)
            path.addLine(to: b.centerPx)
            let line = CAShapeLayer()
            line.path = path.cgPath
            line.lineWidth = 2
            line.strokeColor = randomColor().cgColor
            overlay.addSublayer(line)

            let text = String(format: "%.2fm", distance)
            let mid = CGPoint(x: (a.centerPx.x + b.centerPx.x)/2,
                              y: (a.centerPx.y + b.centerPx.y)/2)
            let tag = CATextLayer()
            tag.contentsScale = UIScreen.main.scale
            tag.fontSize = 13
            tag.alignmentMode = .center
            tag.foregroundColor = UIColor.black.cgColor
            tag.backgroundColor = UIColor.systemYellow.withAlphaComponent(0.9).cgColor
            tag.cornerRadius = 3
            let size = (text as NSString).size(withAttributes: [.font: UIFont.systemFont(ofSize: 13)])
            tag.string = text
            tag.frame = CGRect(x: mid.x - size.width/2 - 4,
                               y: mid.y - size.height/2 - 2,
                               width: size.width + 8,
                               height: size.height + 4)
            overlay.addSublayer(tag)
        }

        // MARK: - 平面标记（水平面：地面→ground，其它→plane；near 回退视作 plane）
        private func markHorizontalPlane(pointWorld: simd_float3) -> String? {
            let planes = planeMap.values.filter { $0.alignment == .horizontal }
            guard !planes.isEmpty else { return nil }

            var bestNear: (ARPlaneAnchor, Float)? = nil

            for p in planes {
                let local = worldToLocal(pointWorld, anchor: p)
                let verticalDist = abs(local.y)

                // 记录最近高度（用于 near 回退）
                if bestNear == nil || verticalDist < bestNear!.1 {
                    bestNear = (p, verticalDist)
                }

                // 高度在阈值内 + 投影点在多边形内 → 命中
                if verticalDist <= planeHeightThreshM, isPoint(local, inside: p) {
                    return isFloor(p) ? "ground" : "plane"
                }
            }

            // near 回退：足够接近也算命中
            if let (p, d) = bestNear, d <= planeNearFallbackM {
                return isFloor(p) ? "ground" : "plane"
            }
            return nil
        }

        // ★ Raycast 兜底：从屏幕 support 像素直接打射线选最近的水平平面
        private func markPlaneViaRaycast(screenPt: CGPoint) -> String? {
            guard let arView else { return nil }

            let queries: [ARRaycastQuery?] = [
                arView.raycastQuery(from: screenPt, allowing: .existingPlaneGeometry, alignment: .horizontal),
                arView.raycastQuery(from: screenPt, allowing: .existingPlaneInfinite,  alignment: .horizontal),
                arView.raycastQuery(from: screenPt, allowing: .estimatedPlane,        alignment: .horizontal)
            ]

            for q in queries.compactMap({ $0 }) {
                if let hit = arView.session.raycast(q).first {
                    if let plane = hit.anchor as? ARPlaneAnchor {
                        return isFloor(plane) ? "ground" : "plane"
                    } else {
                        // 命中的是估计面，仍视作 plane
                        return "plane"
                    }
                }
            }
            return nil
        }

        // —— 为“可视化最近边/轮廓”挑选水平 ARPlaneAnchor（含 raycast 回退）
        private func pickHorizontalAnchorForSupport(worldSupport: simd_float3,
                                                    supportPx: CGPoint) -> ARPlaneAnchor? {
            // 1) 优先：已有水平平面且足够近（高度阈值或 near）
            let planes = planeMap.values.filter { $0.alignment == .horizontal }
            var best: (ARPlaneAnchor, Float)? = nil  // (anchor, verticalDist)
            for p in planes {
                let local = worldToLocal(worldSupport, anchor: p)
                let verticalDist = abs(local.y)
                if verticalDist <= planeNearFallbackM {
                    if best == nil || verticalDist < best!.1 {
                        best = (p, verticalDist)
                    }
                }
            }
            if let b = best { return b.0 }

            // 2) 回退：raycast 命中 existingPlane*(有 anchor) 即用
            guard let arView else { return nil }
            let queries: [ARRaycastQuery?] = [
                arView.raycastQuery(from: supportPx, allowing: .existingPlaneGeometry, alignment: .horizontal),
                arView.raycastQuery(from: supportPx, allowing: .existingPlaneInfinite,  alignment: .horizontal),
            ]
            for q in queries.compactMap({ $0 }) {
                if let hit = arView.session.raycast(q).first,
                   let anchor = hit.anchor as? ARPlaneAnchor {
                    return anchor
                }
            }
            // estimatedPlane 无边界，返回 nil
            return nil
        }

        // —— 计算“最近边”并把轮廓与该边投影到屏幕像素，同时返回该边世界坐标（两端）
        private func nearestEdgeAndOutlineScreen(anchor: ARPlaneAnchor,
                                                 supportWorld: simd_float3)
        -> ((CGPoint, CGPoint)?, [CGPoint]?, (simd_float3, simd_float3)?) {
            guard let arView, let frame = arView.session.currentFrame else { return (nil, nil, nil) }

            // 顶点：优先 boundary，缺省用 extent 矩形
            let vs3 = anchor.geometry.boundaryVertices
            let localXZ: [simd_float2]
            if vs3.count >= 3 {
                localXZ = vs3.map { simd_float2($0.x, $0.z) }
            } else {
                let ex = anchor.extent
                let half = simd_float2(ex.x * 0.5, ex.z * 0.5)
                localXZ = [
                    simd_float2(-half.x, -half.y),
                    simd_float2( half.x, -half.y),
                    simd_float2( half.x,  half.y),
                    simd_float2(-half.x,  half.y)
                ]
            }

            // support 的局部坐标
            let lp = worldToLocal(supportWorld, anchor: anchor)
            let p2 = simd_float2(lp.x, lp.z)

            // 找最近边
            var best: (simd_float2, simd_float2, Float)? = nil
            let n = localXZ.count
            for i in 0..<n {
                let a = localXZ[i]
                let b = localXZ[(i+1) % n]
                let d = pointToSegmentDistance2D(p2, a, b)
                if best == nil || d < best!.2 { best = (a, b, d) }
            }

            // 局部 → 世界 → 屏幕
            func local2world(_ v: simd_float2) -> simd_float3 {
                let T = anchor.transform
                let p = simd_float4(v.x, 0, v.y, 1)
                let w = T * p
                return simd_float3(w.x, w.y, w.z)
            }
            func world2screen(_ w: simd_float3) -> CGPoint? {
                let proj = frame.camera.projectPoint(w,
                                                     orientation: currentInterfaceOrientation(of: arView),
                                                     viewportSize: arView.bounds.size)
                if proj.x.isFinite && proj.y.isFinite { return proj } else { return nil }
            }

            // 轮廓
            var outlinePx: [CGPoint] = []
            for v in localXZ {
                if let sp = world2screen(local2world(v)) { outlinePx.append(sp) }
            }

            // 最近边
            var edgePx: (CGPoint, CGPoint)? = nil
            var edgeWorld: (simd_float3, simd_float3)? = nil
            if let best {
                let A = local2world(best.0), B = local2world(best.1)
                edgeWorld = (A, B)
                if let As = world2screen(A), let Bs = world2screen(B) { edgePx = (As, Bs) }
            }

            return (edgePx, outlinePx.count >= 3 ? outlinePx : nil, edgeWorld)
        }

        private func isFloor(_ anchor: ARPlaneAnchor) -> Bool {
            if #available(iOS 13.0, *) {
                return anchor.classification == .floor
            } else { return false }
        }

        private func worldToLocal(_ pWorld: simd_float3, anchor: ARPlaneAnchor) -> simd_float3 {
            let inv = simd_inverse(anchor.transform)
            let pw = simd_float4(pWorld.x, pWorld.y, pWorld.z, 1)
            let pl = inv * pw
            return simd_float3(pl.x, pl.y, pl.z)
        }

        // 点是否落在 plane 的 boundary polygon 内（在锚点局部坐标系下测试 xz）
        private func isPoint(_ local: simd_float3, inside anchor: ARPlaneAnchor) -> Bool {
            let verts = anchor.geometry.boundaryVertices
            guard verts.count >= 3 else { return false }
            // 射线奇偶法：在 xz 平面做多边形包含测试
            var inside = false
            var j = verts.count - 1
            for i in 0..<verts.count {
                let vi = verts[i]
                let vj = verts[j]
                let intersect = ((vi.z > local.z) != (vj.z > local.z)) &&
                                (local.x < (vj.x - vi.x) * (local.z - vi.z) / ((vj.z - vi.z) + 1e-6) + vi.x)
                if intersect { inside.toggle() }
                j = i
            }
            return inside
        }

        // MARK: - 2D 距离工具（xz 平面）
        private func pointToSegmentDistance2D(_ p: simd_float2,
                                              _ a: simd_float2,
                                              _ b: simd_float2) -> Float {
            let ab = b - a
            let ap = p - a
            let ab2 = max(simd_dot(ab, ab), 1e-12)
            let t = simd_clamp(simd_dot(ap, ab) / ab2, 0, 1)
            let proj = a + t * ab
            return simd_length(p - proj)
        }

        // MARK: - 工具函数（核心：多点鲁棒采样）

        // 多点鲁棒采样：返回（内点世界坐标中位数，欧氏距离平均/回退中位数）
        private func robustWorldAndDistanceAround(center: CGPoint,
                                                  frame: ARFrame,
                                                  viewSize: CGSize,
                                                  depthPB: CVPixelBuffer,
                                                  confPB: CVPixelBuffer?) -> (world: simd_float3, distance: Float)? {
            var worlds: [simd_float3] = []
            var dists:  [Float] = []

            let camT = frame.camera.transform
            let camPos = simd_float3(camT.columns.3.x, camT.columns.3.y, camT.columns.3.z)

            for gy in -sampleGridHalf...sampleGridHalf {
                for gx in -sampleGridHalf...sampleGridHalf {
                    let p = CGPoint(x: center.x + CGFloat(gx)*sampleStepPx,
                                    y: center.y + CGFloat(gy)*sampleStepPx)
                    if let w = worldAtSingle(screenPt: p,
                                             frame: frame,
                                             viewSize: viewSize,
                                             depthPB: depthPB,
                                             confPB: confPB) {
                        worlds.append(w)
                        dists.append(simd_length(w - camPos))
                    }
                }
            }
            guard !dists.isEmpty else { return nil }

            // 以“距离中位数”为基准做离群剔除
            let med = median(of: dists)
            let maxOffset = max(outlierAbsThreshM, outlierRelThresh * med)
            var inlierWorlds: [simd_float3] = []
            var inlierDists:  [Float] = []
            for (w, d) in zip(worlds, dists) {
                if abs(d - med) <= maxOffset {
                    inlierWorlds.append(w)
                    inlierDists.append(d)
                }
            }
            if inlierDists.count < minInliersForAvg {
                // 内点不足，回退使用中位数世界坐标 + 中位数距离
                let wMed = medianWorld(of: worlds)
                return (wMed, med)
            } else {
                // 世界坐标用分量中位数，距离取内点平均
                let wMed = medianWorld(of: inlierWorlds)
                let avgD = inlierDists.reduce(0, +) / Float(inlierDists.count)
                return (wMed, avgD)
            }
        }

        // 对任意屏幕点做鲁棒采样（备用）
        private func robustWorldAt(screenPt: CGPoint,
                                   frame: ARFrame,
                                   viewSize: CGSize,
                                   depthPB: CVPixelBuffer,
                                   confPB: CVPixelBuffer?) -> simd_float3? {
            var worlds: [simd_float3] = []
            for gy in -sampleGridHalf...sampleGridHalf {
                for gx in -sampleGridHalf...sampleGridHalf {
                    let p = CGPoint(x: screenPt.x + CGFloat(gx)*sampleStepPx,
                                    y: screenPt.y + CGFloat(gy)*sampleStepPx)
                    if let w = worldAtSingle(screenPt: p,
                                             frame: frame,
                                             viewSize: viewSize,
                                             depthPB: depthPB,
                                             confPB: confPB) {
                        worlds.append(w)
                    }
                }
            }
            guard !worlds.isEmpty else { return nil }
            return medianWorld(of: worlds)
        }

        // 单点 → 世界点（深度中位数+置信度过滤+坐标映射）
        private func worldAtSingle(screenPt: CGPoint,
                                   frame: ARFrame,
                                   viewSize: CGSize,
                                   depthPB: CVPixelBuffer,
                                   confPB: CVPixelBuffer?) -> simd_float3? {
            guard let (du, dv, imgUV) = mapScreenPointToDepthUV(frame: frame,
                                                                screenPt: screenPt,
                                                                viewSize: viewSize,
                                                                depthPB: depthPB),
                  let md = medianDepth(depthPB: depthPB,
                                       confPB: confPB,
                                       u: du, v: dv,
                                       minConf: confidenceThreshold,
                                       win: medianWindow) else { return nil }

            let z = md.z
            let imgW = Float(frame.camera.imageResolution.width)
            let imgH = Float(frame.camera.imageResolution.height)
            let u_img_px = Float(imgUV.x) * imgW
            let v_img_px = Float(imgUV.y) * imgH
            return backProjectToWorld(u_img_px: u_img_px,
                                      v_img_px: v_img_px,
                                      z: z,
                                      intrinsics: frame.camera.intrinsics,
                                      camTransform: frame.camera.transform)
        }

        // 中位数（Float）
        private func median(of arr: [Float]) -> Float {
            let s = arr.sorted()
            return s[s.count/2]
        }

        // 分量中位数的世界坐标
        private func medianWorld(of arr: [simd_float3]) -> simd_float3 {
            let xs = arr.map { $0.x }.sorted()
            let ys = arr.map { $0.y }.sorted()
            let zs = arr.map { $0.z }.sorted()
            let i = arr.count/2
            return simd_float3(xs[i], ys[i], zs[i])
        }

        //通用工具

        // 当前界面朝向（用于 Vision 和 displayTransform/projectPoint）
        private func currentInterfaceOrientation(of view: UIView) -> UIInterfaceOrientation {
            view.window?.windowScene?.interfaceOrientation ?? .portrait
        }

        // UIInterfaceOrientation -> CGImagePropertyOrientation（供 Vision）
        private func cgImageOrientation(for io: UIInterfaceOrientation) -> CGImagePropertyOrientation {
            switch io {
            case .portrait:             return .right
            case .portraitUpsideDown:   return .left
            case .landscapeLeft:        return .up
            case .landscapeRight:       return .down
            default:                    return .right
            }
        }

        // VN 归一化框（原点左下）→ 屏幕像素（原点左上）
        private func vnRectToPixel(bbox: CGRect, in viewSize: CGSize) -> CGRect {
            let x = bbox.origin.x * viewSize.width
            let w = bbox.size.width * viewSize.width
            let y = (1.0 - (bbox.origin.y + bbox.size.height)) * viewSize.height
            let h = bbox.size.height * viewSize.height
            return CGRect(x: x, y: y, width: w, height: h)
        }

        // 屏幕像素点 -> (深度像素 du,dv) + (相机图像归一化坐标 imgUV)
        // 用“当前界面朝向”的逆变换做 View→Image 映射
        private func mapScreenPointToDepthUV(frame: ARFrame,
                                             screenPt: CGPoint,
                                             viewSize: CGSize,
                                             depthPB: CVPixelBuffer)
        -> (du: Int, dv: Int, imgUV: CGPoint)? {
            let screenUV = CGPoint(x: screenPt.x / viewSize.width,
                                   y: screenPt.y / viewSize.height)

            let io = currentInterfaceOrientation(of: arView!)
            let t_imageToView = frame.displayTransform(for: io, viewportSize: viewSize)
            let t_viewToImage = t_imageToView.inverted()
            let imgUV = CGPointApplyAffineTransform(screenUV, t_viewToImage)

            let dw = CVPixelBufferGetWidth(depthPB)
            let dh = CVPixelBufferGetHeight(depthPB)
            let du = Int(round(CGFloat(dw) * imgUV.x))
            let dv = Int(round(CGFloat(dh) * imgUV.y))

            if du < 0 || dv < 0 || du >= dw || dv >= dh { return nil }
            return (du, dv, imgUV)
        }

        // 取深度置信度（0=低,1=中,2=高）
        private func sampleConfidence(confPB: CVPixelBuffer, u: Int, v: Int) -> UInt8? {
            CVPixelBufferLockBaseAddress(confPB, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(confPB, .readOnly) }
            guard let base = CVPixelBufferGetBaseAddress(confPB) else { return nil }
            let rowBytes = CVPixelBufferGetBytesPerRow(confPB)
            let stride = rowBytes / MemoryLayout<UInt8>.size
            let ptr = base.assumingMemoryBound(to: UInt8.self)
            return ptr[v * stride + u]
        }

        // 中位数深度（win×win），过滤 0/NaN 与低置信像素；返回 (深度, 有效采样数)
        private func medianDepth(depthPB: CVPixelBuffer,
                                 confPB: CVPixelBuffer?,
                                 u: Int, v: Int,
                                 minConf: UInt8,
                                 win: Int = 7) -> (z: Float, count: Int)? {
            CVPixelBufferLockBaseAddress(depthPB, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(depthPB, .readOnly) }

            let w = CVPixelBufferGetWidth(depthPB)
            let h = CVPixelBufferGetHeight(depthPB)
            let rowBytes = CVPixelBufferGetBytesPerRow(depthPB)
            guard let base = CVPixelBufferGetBaseAddress(depthPB) else { return nil }
            let ptr = base.assumingMemoryBound(to: Float32.self)
            let stride = rowBytes / MemoryLayout<Float32>.size

            var vals: [Float] = []
            let r = max(1, win/2)
            for j in max(0, v-r)...min(h-1, v+r) {
                for i in max(0, u-r)...min(w-1, u+r) {
                    if let confPB,
                       let c = sampleConfidence(confPB: confPB, u: i, v: j),
                       c < minConf { continue }
                    let z = ptr[j*stride + i]
                    if z.isFinite && z > 0 { vals.append(z) }
                }
            }
            guard !vals.isEmpty else { return nil }
            vals.sort()
            return (vals[vals.count/2], vals.count)
        }

        // 用相机图像像素坐标 + intrinsics 反投影到世界
        private func backProjectToWorld(u_img_px: Float, v_img_px: Float, z: Float,
                                        intrinsics: simd_float3x3,
                                        camTransform: simd_float4x4) -> simd_float3 {
            let fx = intrinsics[0,0], fy = intrinsics[1,1]
            let cx = intrinsics[2,0], cy = intrinsics[2,1]
            let xNorm = (u_img_px - cx) / fx
            let yNorm = (v_img_px - cy) / fy
            let Xc = xNorm * z, Yc = yNorm * z, Zc = z
            let camP = simd_float4(Xc, Yc, Zc, 1)
            let worldP = camTransform * camP
            return simd_float3(worldP.x, worldP.y, worldP.z)
        }

        // Raycast 回退（备用，可用于远距容错）
        private func fallbackDepthViaRaycast(screenPt: CGPoint,
                                             view: ARSCNView,
                                             frame: ARFrame) -> (world: simd_float3, distance: Float)? {
            let queries: [ARRaycastQuery?] = [
                view.raycastQuery(from: screenPt, allowing: .existingPlaneGeometry, alignment: .any),
                view.raycastQuery(from: screenPt, allowing: .existingPlaneInfinite, alignment: .any),
                view.raycastQuery(from: screenPt, allowing: .estimatedPlane, alignment: .any)
            ]
            for q in queries.compactMap({ $0 }) {
                if let first = view.session.raycast(q).first {
                    let wp = first.worldTransform.columns.3
                    let world = simd_float3(wp.x, wp.y, wp.z)
                    let camT = frame.camera.transform
                    let camPos = simd_float3(camT.columns.3.x, camT.columns.3.y, camT.columns.3.z)
                    let dist = simd_length(world - camPos)
                    if dist > 8.0 { continue } // 远距离误命中过滤
                    return (world, dist)
                }
            }
            return nil
        }

        // 随机颜色
        private func randomColor() -> UIColor {
            UIColor(red: CGFloat.random(in: 0.25...1),
                    green: CGFloat.random(in: 0.25...1),
                    blue: CGFloat.random(in: 0.25...1),
                    alpha: 0.95)
        }
    }
}

// MARK: - 小工具
@inline(__always)
func CGPointApplyAffineTransform(_ p: CGPoint, _ t: CGAffineTransform) -> CGPoint {
    CGPoint(x: p.x * t.a + p.y * t.c + t.tx,
            y: p.x * t.b + p.y * t.d + t.ty)
}

