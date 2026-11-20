# Sourced by setup/build; the parent gitlinks must agree with these pins.
WEBRTC_REVISION=95554e4a0c5c343fdc989387d9d0ce18e2a2294b
DEPOT_TOOLS_REVISION=ec77017fb3e090ff02bffb9c6980ebcdd3006ddc

# Exactly this tracked patch is allowed on the pinned WebRTC checkout. Reject
# other local edits rather than resetting or silently building different sources.
webrtc_compat_is_applied() {
  git -C "$ROOT/third_party/webrtc/src" diff --no-ext-diff --binary HEAD |
    cmp -s "$ROOT/config/webrtc/libstdcxx11.patch" -
}
