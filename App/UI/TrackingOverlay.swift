import SwiftUI
import AspectusKit

/// draws the tracked face box, eyes, and pupils over the preview for visual verification
/// replicates the renderer's aspect-fill and mirror transform so geometry lines up with pixels
struct TrackingOverlay: View {
    @ObservedObject var controller: PipelineController

    var body: some View {
        GeometryReader { geo in
            Canvas { ctx, size in
                guard controller.showOverlay,
                      let tr = controller.tracking,
                      controller.imageWidth > 0, controller.imageHeight > 0 else { return }

                let map = Mapper(viewSize: size,
                                 imageW: Double(controller.imageWidth),
                                 imageH: Double(controller.imageHeight),
                                 mirror: controller.mirrorPreview)

                stroke(rect: tr.faceBounds, in: &ctx, map: map,
                       color: .green.opacity(0.7), width: 1.5)
                stroke(rect: tr.leftEye.region, in: &ctx, map: map, color: .cyan, width: 1.5)
                stroke(rect: tr.rightEye.region, in: &ctx, map: map, color: .cyan, width: 1.5)
                dot(at: tr.leftEye.pupilCenter, in: &ctx, map: map)
                dot(at: tr.rightEye.pupilCenter, in: &ctx, map: map)
            }
            .allowsHitTesting(false)
        }
    }

    private func stroke(rect r: NormRect, in ctx: inout GraphicsContext,
                        map: Mapper, color: Color, width: Double) {
        // bound the mapped corners since mirroring flips left/right
        let p1 = map.point(r.x, r.y)
        let p2 = map.point(r.x + r.width, r.y + r.height)
        let rect = CGRect(x: min(p1.x, p2.x), y: min(p1.y, p2.y),
                          width: abs(p2.x - p1.x), height: abs(p2.y - p1.y))
        ctx.stroke(Path(rect), with: .color(color), lineWidth: width)
    }

    private func dot(at p: NormPoint, in ctx: inout GraphicsContext, map: Mapper) {
        let c = map.point(p.x, p.y)
        let r = 3.0
        ctx.fill(Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r)),
                 with: .color(.yellow))
    }

    /// inverse of the shader uv transform, image-normalized top-left to view pixel
    struct Mapper {
        let sx: Double, sy: Double
        let viewSize: CGSize
        let mirror: Bool

        init(viewSize: CGSize, imageW: Double, imageH: Double, mirror: Bool) {
            self.viewSize = viewSize
            self.mirror = mirror
            let texAspect = imageW / imageH
            let viewAspect = viewSize.width / max(1, viewSize.height)
            if texAspect > viewAspect {
                sx = viewAspect / texAspect; sy = 1
            } else {
                sx = 1; sy = texAspect / viewAspect
            }
        }

        func point(_ ix: Double, _ iy: Double) -> CGPoint {
            var ux = (ix - 0.5) / sx + 0.5
            let uy = (iy - 0.5) / sy + 0.5
            if mirror { ux = 1 - ux }
            return CGPoint(x: ux * viewSize.width, y: uy * viewSize.height)
        }
    }
}
