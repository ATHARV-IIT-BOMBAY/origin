// gestures.js — pure hand-landmark math for HoloControl.
// No DOM, no three.js: just geometry on MediaPipe hand landmarks, so it's unit-testable.
// A "hand" is an array of 21 landmarks, each { x, y, z } in normalized [0,1] image coords.

// MediaPipe hand landmark indices we care about.
export const WRIST = 0;
export const THUMB_TIP = 4;
export const INDEX_TIP = 8;
export const MIDDLE_MCP = 9; // stable palm anchor (middle-finger knuckle)

const PALM_POINTS = [0, 5, 9, 13, 17]; // wrist + the four finger bases = the palm

export function dist2d(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

// Palm size = wrist -> middle-knuckle distance. We normalize other distances by this
// so a gesture means the same thing whether the hand is near or far from the camera.
export function palmSize(hand) {
  return dist2d(hand[WRIST], hand[MIDDLE_MCP]) || 1e-6;
}

// Centroid of the palm points — a steadier "hand position" than any single jittery tip.
export function handCenter(hand) {
  let x = 0, y = 0;
  for (const i of PALM_POINTS) { x += hand[i].x; y += hand[i].y; }
  return { x: x / PALM_POINTS.length, y: y / PALM_POINTS.length };
}

// 0 = fingers open, 1 = thumb and index touching. Scale-invariant (divided by palm size).
export function pinchStrength(hand) {
  const d = dist2d(hand[THUMB_TIP], hand[INDEX_TIP]) / palmSize(hand);
  const PINCH_MIN = 0.25, PINCH_MAX = 0.9; // d ~0.1 when pinched, ~1.0 when open
  return Math.max(0, Math.min(1, (PINCH_MAX - d) / (PINCH_MAX - PINCH_MIN)));
}

export function isPinching(hand, threshold = 0.6) {
  return pinchStrength(hand) >= threshold;
}

// Distance between two hands' centers — drives zoom (spread apart = zoom in).
export function twoHandSpread(handA, handB) {
  return dist2d(handCenter(handA), handCenter(handB));
}

// Signed angle (radians) of the line joining the two hand centers. Twisting both hands
// like a steering wheel rotates this line; the frame-to-frame change drives roll (rotation
// about the view axis). Callers must unwrap the delta across the ±π seam themselves.
export function twoHandAngle(handA, handB) {
  const a = handCenter(handA), b = handCenter(handB);
  return Math.atan2(b.y - a.y, b.x - a.x);
}

// Per-frame roll (radians) from a two-hand twist. `zooming` (both hands pinched) returns 0 so a
// resize stays a pure scale and axis locks actually hold during a zoom. Otherwise roll by the
// unwrapped angle change, dropping sub-`deadzone` wobble so only a deliberate twist rolls. `mirror`
// is ±1 to match the mirrored preview. prevAngle == null (first frame) also returns 0.
export function rollDelta(prevAngle, ang, { zooming = false, deadzone = 0.012, mirror = -1 } = {}) {
  if (prevAngle == null || zooming) return 0;
  let d = ang - prevAngle;
  d = Math.atan2(Math.sin(d), Math.cos(d));   // unwrap across the ±π seam
  return Math.abs(d) > deadzone ? d * mirror : 0;
}

// Map a normalized image landmark ({x,y in [0,1]}, z ~ relative depth) into three.js
// world space, so we can draw the hand floating inside the scene. Mirrored on X to match
// the mirrored webcam preview: move your real hand right, the on-screen hand goes right.
export function landmarkToWorld(pt, span = 4, depth = 1.5) {
  return {
    x: -(pt.x - 0.5) * span,
    y: -(pt.y - 0.5) * span,
    z: 1.2 - (pt.z || 0) * depth,
  };
}

// A finger is "extended" when its tip reaches noticeably farther from the wrist than its
// middle (PIP) joint — i.e. straightened, not curled back toward the palm. Rotation-robust
// because it compares distances, not absolute up/down. (tip, pip) are landmark indices.
export function fingerExtended(hand, tip, pip) {
  return dist2d(hand[tip], hand[WRIST]) > dist2d(hand[pip], hand[WRIST]) * 1.15;
}

// Classify the hand into the one pose we act on: 'fist' | 'pinch' | 'point' | 'open'.
// fist = every finger curled (used to "park"/ignore a hand), pinch = thumb+index together
// (grab), point = only the index finger out (aim), else open. Fist is tested before pinch:
// a tight fist tucks the thumb against the index and would otherwise misread as a pinch.
export function handPose(hand) {
  const idx = fingerExtended(hand, 8, 6);
  const mid = fingerExtended(hand, 12, 10);
  const rng = fingerExtended(hand, 16, 14);
  const pky = fingerExtended(hand, 20, 18);
  if (!idx && !mid && !rng && !pky) return 'fist';
  if (pinchStrength(hand) >= 0.6) return 'pinch';
  return (idx && !mid && !rng && !pky) ? 'point' : 'open';
}

// Where to drop a freshly-loaded model so it sits centered on the holo-stage. Uploaded models
// carry an arbitrary origin — often far from the geometry — so naively they land "at a random
// point". Given the model's bounding box (measured at scale 1) we return the {scale, position}
// that: scales the largest side to `target`, centers it on X/Z, and rests its bottom on `floorY`
// (feet on the floor). ORDER MATTERS: the final world box is `position + scale*localBox`, so we
// solve position AFTER scaling — centering before scaling (the old bug) re-adds center*(scale-1).
export function fitTransform(min, max, target = 1.8, floorY = -1.55) {
  const sx = max.x - min.x, sy = max.y - min.y, sz = max.z - min.z;
  const scale = target / (Math.max(sx, sy, sz) || 1);
  return {
    scale,
    position: {
      x: -scale * (min.x + max.x) / 2, // scaled box centered on X
      y: floorY - scale * min.y,       // scaled box bottom sits on the floor
      z: -scale * (min.z + max.z) / 2, // ...and on Z
    },
  };
}

