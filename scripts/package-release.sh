#!/bin/bash
#
# Builds, signs, notarizes and packages an Aspectus release for GitHub.
#
# No secret is ever passed to this script or stored in the repository. Notarization credentials
# live in the login keychain under a named profile that you create once, yourself:
#
#     xcrun notarytool store-credentials "$NOTARY_PROFILE" \
#         --apple-id <your-apple-id> --team-id <your-team-id> --password <app-specific-password>
#
# The script refuses to produce a "release" it cannot honestly label. If signing or notarization
# is unavailable it either stops, or — with --allow-unsigned — produces an artifact that is named
# and documented as unsigned, never as notarized.

set -euo pipefail

PROJECT="Aspectus.xcodeproj"
SCHEME="Aspectus"
CONFIG="Release"
BUILD_DIR=".build/release-package"
NOTARY_PROFILE="${NOTARY_PROFILE:-aspectus-notary}"
ALLOW_UNSIGNED=0

for arg in "$@"; do
  case "$arg" in
    --allow-unsigned) ALLOW_UNSIGNED=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
fail() { printf '\033[31merror: %s\033[0m\n' "$1" >&2; exit 1; }

# ---------------------------------------------------------------- preconditions

[ -f "Signing.xcconfig" ] || fail "Signing.xcconfig is missing. Copy Signing.xcconfig.example and set your team."
TEAM_ID=$(sed -n 's/^ASPECTUS_TEAM_ID *= *//p' Signing.xcconfig | tr -d ' ')

IDENTITY=$(security find-identity -v -p codesigning | grep "Developer ID Application" | head -1 || true)
if [ -z "$IDENTITY" ]; then
  if [ "$ALLOW_UNSIGNED" -eq 0 ]; then
    fail "no Developer ID Application certificate found.

A notarized release requires one. Create it as the Account Holder of your Developer Program team:
  Xcode > Settings > Accounts > select the team > Manage Certificates > + > Developer ID Application

To build an explicitly unsigned artifact for testing instead, re-run with --allow-unsigned."
  fi
  say "NO Developer ID certificate — building an UNSIGNED artifact"
else
  say "signing identity: $(echo "$IDENTITY" | sed 's/.*"\(.*\)"/\1/')"
fi

VERSION=$(sed -n 's/.*MARKETING_VERSION: *"\(.*\)"/\1/p' project.yml | head -1)
COMMIT=$(git rev-parse --short HEAD)
DIRTY=$(git status --porcelain | wc -l | tr -d ' ')
[ "$DIRTY" -eq 0 ] || fail "the working tree has uncommitted changes; a release must be reproducible from a commit"

say "Aspectus $VERSION ($COMMIT), team ${TEAM_ID:-none}"

# ---------------------------------------------------------------- build

say "generating the project"
xcodegen generate >/dev/null

rm -rf "$BUILD_DIR"
if [ -n "$IDENTITY" ]; then
  # a Developer ID build cannot come from `xcodebuild build`: automatic signing there always
  # resolves to a development profile, which conflicts with the Developer ID identity. archiving
  # and exporting with method=developer-id is the path that provisions the System Extension
  # capability correctly
  say "archiving $CONFIG for Apple Silicon"
  ARCHIVE="$BUILD_DIR/Aspectus.xcarchive"
  xcodebuild -project "$PROJECT" -scheme "$SCHEME" -configuration "$CONFIG" \
    -derivedDataPath "$BUILD_DIR" -archivePath "$ARCHIVE" -arch arm64 \
    -allowProvisioningUpdates archive >/dev/null

  say "exporting with Developer ID"
  cat > "$BUILD_DIR/ExportOptions.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>method</key><string>developer-id</string>
	<key>teamID</key><string>$TEAM_ID</string>
	<key>signingStyle</key><string>automatic</string>
	<key>destination</key><string>export</string>
</dict>
</plist>
PLIST
  xcodebuild -exportArchive -archivePath "$ARCHIVE" \
    -exportOptionsPlist "$BUILD_DIR/ExportOptions.plist" \
    -exportPath "$BUILD_DIR/export" -allowProvisioningUpdates >/dev/null
  APP="$BUILD_DIR/export/Aspectus.app"
else
  say "building $CONFIG unsigned"
  xcodebuild -project "$PROJECT" -scheme "$SCHEME" -configuration "$CONFIG" \
    -derivedDataPath "$BUILD_DIR" -arch arm64 \
    CODE_SIGNING_ALLOWED=NO CODE_SIGNING_REQUIRED=NO ENTITLEMENTS_REQUIRED=NO build >/dev/null
  APP="$BUILD_DIR/Build/Products/$CONFIG/Aspectus.app"
fi

[ -d "$APP" ] || fail "the build produced no app bundle"

# ---------------------------------------------------------------- verify contents

say "verifying the bundle"
EXT="$APP/Contents/Library/SystemExtensions/com.aspectus.app.cameraextension.systemextension"
[ -d "$EXT" ] || fail "the camera extension is not embedded; the release would be a preview-only app"
echo "  camera extension embedded"

ARCHS=$(lipo -archs "$APP/Contents/MacOS/Aspectus")
echo "  architectures: $ARCHS"
case "$ARCHS" in *arm64*) ;; *) fail "the binary is not arm64" ;; esac

if [ -n "$IDENTITY" ]; then
  codesign --verify --deep --strict --verbose=2 "$APP" 2>&1 | sed 's/^/  /'
  codesign -d --entitlements - --xml "$APP" 2>/dev/null | plutil -p - | sed 's/^/  /'
fi

# ---------------------------------------------------------------- notarize

STAPLED=0
if [ -n "$IDENTITY" ]; then
  if xcrun notarytool history --keychain-profile "$NOTARY_PROFILE" >/dev/null 2>&1; then
    say "notarizing (this waits for Apple and can take several minutes)"
    ZIP_FOR_NOTARY="$BUILD_DIR/Aspectus-notary.zip"
    ditto -c -k --keepParent "$APP" "$ZIP_FOR_NOTARY"
    xcrun notarytool submit "$ZIP_FOR_NOTARY" --keychain-profile "$NOTARY_PROFILE" --wait
    say "stapling the ticket"
    xcrun stapler staple "$APP"
    xcrun stapler validate "$APP" && STAPLED=1
  else
    fail "no notary credentials in the keychain under profile '$NOTARY_PROFILE'.

Store them once, yourself — the password never passes through this script:
  xcrun notarytool store-credentials \"$NOTARY_PROFILE\" \\
      --apple-id <your-apple-id> --team-id $TEAM_ID --password <app-specific-password>

Create the app-specific password at appleid.apple.com > Sign-In and Security."
  fi
fi

# ---------------------------------------------------------------- package

say "packaging"
OUT="dist"
mkdir -p "$OUT"
if [ "$STAPLED" -eq 1 ]; then
  NAME="Aspectus-$VERSION-arm64"
elif [ -n "$IDENTITY" ]; then
  NAME="Aspectus-$VERSION-arm64-signed-not-notarized"
else
  NAME="Aspectus-$VERSION-arm64-UNSIGNED"
fi
ZIP="$OUT/$NAME.zip"
rm -f "$ZIP"
ditto -c -k --keepParent "$APP" "$ZIP"

shasum -a 256 "$ZIP" | tee "$ZIP.sha256"

cat > "$OUT/$NAME.txt" <<EOF
Aspectus $VERSION
commit:        $COMMIT
architecture:  arm64 (Apple Silicon only)
extension:     embedded
signing:       $([ -n "$IDENTITY" ] && echo "Developer ID Application, team $TEAM_ID" || echo "UNSIGNED")
notarization:  $([ "$STAPLED" -eq 1 ] && echo "notarized and stapled" || echo "NOT notarized")
sha256:        $(shasum -a 256 "$ZIP" | cut -d' ' -f1)
EOF

say "done"
cat "$OUT/$NAME.txt"

if [ "$STAPLED" -eq 0 ]; then
  printf '\n\033[33mThis artifact is NOT notarized. Do not describe it as notarized.\033[0m\n'
  printf '\033[33mGatekeeper will warn on first launch and users must right-click > Open.\033[0m\n'
fi
