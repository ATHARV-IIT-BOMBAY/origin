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
// unwrapped angle change, band-passed: below `deadzone` is sensor wobble, above `maxStep` is not a
// hand. The upper guard matters because the tracker hands us two hands in no guaranteed order — if
// it swaps them between frames the joining line flips ~180° and the model would snap-roll. No wrist
// twists 20°/frame (~600°/s at 30fps), so rejecting those costs nothing real. `mirror` is ±1 to
// match the mirrored preview. prevAngle == null (first frame) also returns 0.
export function rollDelta(prevAngle, ang, { zooming = false, deadzone = 0.012, maxStep = 0.35, mirror = -1 } = {}) {
  if (prevAngle == null || zooming) return 0;
  let d = ang - prevAngle;
  d = Math.atan2(Math.sin(d), Math.cos(d));   // unwrap across the ±π seam
  const a = Math.abs(d);
  return (a > deadzone && a <= maxStep) ? d * mirror : 0;
}

// Part removal as a pure state machine, so the timing is testable against a fake clock, not a webcam.
// Point at a part and HOLD it: for the first few seconds you're just INSPECTING it (info shows, part
// glows blue). Hold past the dwell and it GRABS — the part turns red and sticks to your fingertip.
// Then move your hand and the part follows, whatever the tracker thinks your pose is: this is the whole
// fix for "it won't move" — MediaPipe's finger classification flickers off 'point' the instant the hand
// moves, so requiring 'point' every frame dropped the grab the moment you tried to drag. We keep the
// grab while a single controlling hand is present (`present`) and only end it two ways: drag the part
// into the dustbin (`inBin`) -> removed; or let go — hand gone, fist, or a second hand — -> snaps home.
//   phase:  'off' -> 'aim' (inspecting, dwell filling) -> 'grab' (stuck to the finger)
//   action: one-shot edge for sound/HUD — null | 'grab' | 'remove' | 'drop'
// `pose` is null when the hand leaves frame; only a steady 'point' at a real `id` fills the dwell.
// `present` = a single controlling hand is still here (defaults to "a hand is in frame"); `inBin` = the
// grabbed part has been dragged into the dustbin. inBin removes; losing `present` drops it home.
export const AIM_OFF = { phase: 'off', id: null, t0: 0, progress: 0, action: null };
export function aimStep(st, { pose, id, now, present = pose != null, inBin = false }, { dwellMs = 5000 } = {}) {
  if (st.phase === 'grab') {                                  // stuck to the finger; pose may flicker as the hand moves
    if (inBin) return { ...AIM_OFF, id: st.id, action: 'remove' };   // dragged into the dustbin
    if (!present) return { ...AIM_OFF, id: st.id, action: 'drop' };  // hand gone / fist / two hands: snap it home
    return { ...st, action: null };                          // otherwise keep following the finger, whatever the pose reads
  }
  if (pose !== 'point' || id == null) return AIM_OFF;         // hand gone, or not pointing at a part
  if (st.phase !== 'aim' || st.id !== id) return { phase: 'aim', id, t0: now, progress: 0, action: null };
  const progress = Math.min(1, (now - st.t0) / dwellMs);
  return progress >= 1
    ? { phase: 'grab', id, t0: now, progress: 1, action: 'grab' }
    : { ...st, progress, action: null };
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

// The palm as a plane. The wrist and the index/pinky knuckles are three non-collinear points on the
// back of the hand, so their cross product is the palm normal — which is all a cut plane needs. Both
// point and normal come back already in world space (via landmarkToWorld) so the caller can hand
// them straight to a three.js clipping plane; the normal is unit length. Because clipping keeps the
// positive side, turning your palm over flips which half of the model survives, which is the whole
// interaction. A degenerate hand (three collinear points) returns +Z rather than NaN.
export function palmPlane(hand, span = 4, depth = 1.5) {
  const a = landmarkToWorld(hand[WRIST], span, depth);
  const b = landmarkToWorld(hand[5], span, depth);   // index knuckle
  const c = landmarkToWorld(hand[17], span, depth);  // pinky knuckle
  const u = { x: b.x - a.x, y: b.y - a.y, z: b.z - a.z };
  const v = { x: c.x - a.x, y: c.y - a.y, z: c.z - a.z };
  let n = { x: u.y * v.z - u.z * v.y, y: u.z * v.x - u.x * v.z, z: u.x * v.y - u.y * v.x };
  const len = Math.hypot(n.x, n.y, n.z);
  n = len < 1e-9 ? { x: 0, y: 0, z: 1 } : { x: n.x / len, y: n.y / len, z: n.z / len };
  return { point: { x: (a.x + b.x + c.x) / 3, y: (a.y + b.y + c.y) / 3, z: (a.z + b.z + c.z) / 3 }, normal: n };
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

