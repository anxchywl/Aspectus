# Aspectus

[![CI](https://github.com/anxchywl/Aspectus/actions/workflows/ci.yml/badge.svg)](https://github.com/anxchywl/Aspectus/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE)

**Look at the camera while you read your screen.** Aspectus corrects your eye contact in real time
and publishes the result as a standard virtual camera, so Zoom, Meet, Teams, Discord, Slack and OBS
see a corrected feed with no plugin to install.

Native macOS, Apple Silicon only. Swift, SwiftUI, AVFoundation, Metal, Apple Vision and a
CoreMediaIO camera extension. No Electron, no Python runtime.

> [!WARNING]
> **In development.** The pipeline runs end to end and the virtual camera works in Zoom, but the
> other five host apps are untested and correction quality is a geometric baseline, not the target.
> There are no releases yet, and building it yourself needs an Apple Developer ID — macOS will not
> activate an unsigned camera extension. See [Status](#status).

## What it does

- **Redirects gaze** by warping only the eye regions, resampling the original pixels — it never
  synthesizes a face, so identity, glasses, eyelids and lighting survive by construction
- **Falls back cleanly** — below the confidence or angle limit the untouched frame passes through
- **Stays stable** — 1€ filters remove flicker without visible lag, and blinks stay sharp
- **Publishes** to a system-wide virtual camera that other apps see as ordinary hardware

Calibration is optional. The app works out of the box on any supported Mac, and what you change in
Settings — the screen-to-lens angle, the camera, the viewing distance, what the preview shows — is
still there next launch.

## How it works

```
Camera → Face & eye tracking → Gaze estimation → Eye correction
       → Temporal stabilization → Metal compositing → Virtual camera
```

At most one frame is ever in flight: capture hands off through a single-slot, drop-stale box and
counts what it drops. Apple Vision locates the face, eye regions, pupils and head pose off the main
actor; a Metal shader rotates the iris toward the lens; a confidence gate decides how much of that
to blend in. The corrected frame goes to the preview and to the camera extension in the same step.

The virtual camera is an output, never a dependency — if the extension is missing or fails, the
preview and correction carry on untouched.

Correction currently uses a geometric eyeball model rather than a learned warp field. It sits behind
the `EyeCorrector` protocol, so replacing it touches nothing else.

## Quick start

**Requires** Xcode 26+, macOS 14+, Apple Silicon.

```bash
brew install xcodegen
xcodebuild -downloadComponent MetalToolchain   # Xcode 26 ships Metal separately (~690 MB)
```

The core library builds and tests without a camera:

```bash
swift test
```

The app:

```bash
xcodegen generate
open Aspectus.xcodeproj      # ⌘R, then grant camera permission
```

This gives you the app and its live preview, but **not** the virtual camera — that needs a signed
build:

```bash
cp scripts/Signing.xcconfig.example Signing.xcconfig   # add your team; the file is gitignored
./scripts/package-release.sh                           # signs, notarizes, staples
```

Launch the packaged app, click **Install camera**, and approve it once under System Settings ▸
General ▸ Login Items & Extensions ▸ Camera Extensions. Aspectus then appears beside your real
cameras. Full build, signing and release detail: [docs/INFRASTRUCTURE.md](./docs/INFRASTRUCTURE.md).

## Performance

Release build, Apple M3 / macOS 26.6, FaceTime HD at 1280×720, over 3,937 frames:

| Stage | mean | p95 |
|---|---|---|
| Face tracking (Vision) | 19.2 ms | 29.2 ms |
| Eye warp (Metal) | 1.8 ms | 2.4 ms |
| Ingest → present | 45.7 ms | 61.2 ms |
| Capture → present | 89.4 ms | 106.0 ms |

Frames in flight never exceeded 1, with 11 drops in 3,937 frames. A separate **97-minute soak**
published 136,479 frames to the virtual camera, dropped 63 (0.036 %), survived three sleep/wake
cycles — 1.8–2.0 s from wake to frames each time — and ended with resident memory *lower* than it
started. **Two targets are missed:** processing latency is 3× over its 20 ms budget — tracking, not
correction, accounts for it — and 60 FPS is unreachable because this camera caps at 30 at every
format it offers. The tracking figure above is also disputed by two later runs at ~6 ms; all of it
is analysed in [docs/DESIGN.md](./docs/DESIGN.md).

Every number here comes from the app's own CSV recorder in a release build, never an estimate.

## Status

| Phase | State |
|---|---|
| 1 — Capture, Metal preview, latency HUD | ✅ measured on device |
| 2 — Vision tracking: landmarks, pupils, head pose | ✅ running |
| 3 — Geometric eye warp in Metal | ✅ working, over the latency budget |
| 3b — Learned warp field (Core ML) | ⬜ blocked — no licence-clean weights |
| 4 — Temporal quality: filters, gate, slew | ✅ wired and tested |
| 5 — Virtual camera (CoreMediaIO) | ✅ verified in Zoom; five other hosts untested |
| 6 — UI and hardening | ✅ settings, saved preferences, menus, inspector; 97-min soak passed |

Known gaps and everything not yet measured on hardware are listed at the end of
[docs/DESIGN.md](./docs/DESIGN.md).

## Layout

```
Sources/AspectusKit/   framework-free core: backpressure, filters, gate, recovery, metrics
Tests/                 unit tests for the real-time invariants
App/                   capture, Metal render, pipeline, SwiftUI, virtual-camera sink
CameraExtension/       the CoreMediaIO system extension, its own process
Shared/                the format contract, compiled into both
scripts/               signing, notarization, packaging
docs/                  design record and infrastructure
```

## Documentation

| | |
|---|---|
| [docs/DESIGN.md](./docs/DESIGN.md) | research, foundation choice, risks, measured facts, known gaps |
| [docs/INFRASTRUCTURE.md](./docs/INFRASTRUCTURE.md) | build system, targets, entitlements, signing, CI, logs |
| [AGENTS.md](./AGENTS.md) | architecture and coding rules for contributors and AI agents |

## Licence

[Apache-2.0](./LICENSE). Copyright 2026 anxchywl. See [NOTICE](./NOTICE) for attribution: the
correction method is a native reimplementation of BSD-3-licensed work, with no code and no trained
weights taken from it.
