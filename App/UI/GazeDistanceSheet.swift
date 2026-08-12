import SwiftUI
import AspectusKit

/// Collects one physically measured eye-to-lens distance.
///
/// Schema 6 exists because schema 5 took this number from a Settings preference that was typed once
/// and silently reused for every session. The sessions were not all recorded at that distance, so
/// screen labels — which are `atan2(offset_mm, distance)` — disagreed with each other by up to four
/// degrees for the same screen position. This sheet therefore starts empty every time: there is no
/// default, because a default is the defect.
struct GazeDistanceSheet: View {
    enum Stage {
        case opening
        case closing(openingMM: Double)

        var title: String {
            switch self {
            case .opening: return "Measure before starting"
            case .closing: return "Measure again"
            }
        }
    }

    let stage: Stage
    /// live crop side, recorded with the measurement as an auditable cross-check
    let cropSidePixels: Double?
    let onConfirm: (GazeDatasetDistanceMeasurement) -> Void
    let onCancel: () -> Void

    @State private var text = ""
    @State private var instrument = "tape measure"

    private var millimetres: Double? { Double(text.trimmingCharacters(in: .whitespaces)) }

    private var measurement: GazeDatasetDistanceMeasurement? {
        guard let millimetres else { return nil }
        let value = GazeDatasetDistanceMeasurement(millimetres: millimetres,
                                                   instrument: instrument,
                                                   cropSidePixels: cropSidePixels)
        return value.isPlausible ? value : nil
    }

    /// shown while typing rather than on submit, so the participant is not told to measure again
    /// after they have already put the rule away
    private var drift: Double? {
        guard case let .closing(openingMM) = stage, let millimetres else { return nil }
        return abs(millimetres - openingMM)
    }

    private var driftExceeded: Bool {
        (drift ?? 0) > GazeDatasetSchema6.distanceAgreementToleranceMM
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(stage.title).font(.title2.weight(.semibold))

            Text("Measure from your eye to the camera lens with a rule or tape, and type the "
                 + "result. This is the number every screen label is computed from, so it is "
                 + "measured for each session rather than remembered between them.")
                .font(.callout).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            HStack(spacing: 8) {
                TextField("distance", text: $text)
                    .frame(width: 90).multilineTextAlignment(.trailing)
                Text("mm")
                Spacer()
                Text("instrument").foregroundStyle(.secondary)
                TextField("instrument", text: $instrument).frame(width: 130)
            }

            if let cropSidePixels {
                Text(String(format: "crop side now %.1f px — recorded with the measurement as a "
                            + "cross-check, never used to derive it", cropSidePixels))
                    .font(.caption).foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if case let .closing(openingMM) = stage {
                Text(String(format: "measured %.0f mm before starting", openingMM))
                    .font(.caption.monospaced()).foregroundStyle(.secondary)
            }

            if let drift, driftExceeded {
                Label(String(format: "%.0f mm from the opening measurement, past the %.0f mm "
                             + "limit. Recording this ends the session as unusable: no single "
                             + "distance describes it, so its screen labels would be wrong.",
                             drift, GazeDatasetSchema6.distanceAgreementToleranceMM),
                      systemImage: "exclamationmark.triangle.fill")
                    .font(.caption).foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }

            if millimetres != nil && measurement == nil {
                Text("Enter a distance between 150 and 2000 mm, and name the instrument.")
                    .font(.caption).foregroundStyle(.orange)
            }

            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { onCancel() }
                Button(confirmTitle) { if let measurement { onConfirm(measurement) } }
                    .buttonStyle(.borderedProminent)
                    .disabled(measurement == nil)
            }
        }
        .padding(24).frame(width: 470)
    }

    private var confirmTitle: String {
        switch stage {
        case .opening: return "Start session"
        case .closing: return driftExceeded ? "Record and end session" : "Finish session"
        }
    }
}
