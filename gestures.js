// gestures.js — pure hand-landmark math for Origin.
// No DOM, no three.js: just geometry on MediaPipe hand landmarks, so it's unit-testable.
// A "hand" is an array of 21 landmarks, each { x, y, z } in normalized [0,1] image coords.

// MediaPipe hand landmark indices we care about.
export const WRIST = 0;
export const THUMB_TIP = 4;
export const INDEX_TIP = 8;
export const MIDDLE_TIP = 12;
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

// The pinch point: the midpoint of the thumb tip and the index tip. This is where the hand actually
// closes on something — the spot you line up with a part on screen — so it's the right anchor for
// "grab what I'm pinching". Far better than the palm centroid, which sits back toward the wrist,
// offset from where the fingers converge, so pinching AT a part measured from the wrong place.
export function pinchPoint(hand) {
  return { x: (hand[THUMB_TIP].x + hand[INDEX_TIP].x) / 2, y: (hand[THUMB_TIP].y + hand[INDEX_TIP].y) / 2 };
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

// A DELIBERATE pinch, for pose classification — stricter than pinchStrength alone, which fired far too
// readily (a relaxed open palm or a fist both put the thumb within ~0.5 palms of the index and read as a
// pinch — the over-sensitive grab). A real pinch closes the thumb ON THE INDEX SPECIFICALLY: (1) the tips
// genuinely touch (< ~0.4 palms apart), AND (2) the thumb is clearly nearer the index than the middle tip.
// (2) is what rejects the impostors — a fist tucks the thumb near ALL the fingertips (thumb–index ≈ thumb–
// middle), and an open palm holds it near none; only in a true pinch is the index distinctly the closest.
export function isPinchPose(hand) {
  const ti = dist2d(hand[THUMB_TIP], hand[INDEX_TIP]);
  const tm = dist2d(hand[THUMB_TIP], hand[MIDDLE_TIP]);
  return ti / palmSize(hand) < 0.4 && ti < tm * 0.7;   // ponytail: 0.4 / 0.7 tuned to reject palm & fist; loosen if real pinches miss, tighten if palms still grab
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

// Part extraction as a pure state machine, so the timing is testable against fake inputs, not a webcam.
// PROXIMITY + POSE model (the user's design): the caller finds the part nearest the hand's aim point ON
// SCREEN and passes its `id` when it's within reach (null when nothing is close). POINT one finger at a
// piece and it's INSPECTED — the caller times a short dwell before it commits the info card, so a glance
// doesn't fire (blue glow, no commitment). PINCH (thumb+index) while a part is in reach and you GRAB it
// directly — no point or fist first. The part turns red and sticks to your hand 1:1. OPEN your hand to let
// go: over the dustbin it's REMOVED, anywhere else it snaps HOME. Nothing is time-based, so nothing deletes
// on a timer and carrying a part through the bin does nothing — removal is only the deliberate open-over-bin.
// A FIST is free for whole-model rotate (see main.js); an OPEN hand is neutral/ignored — moving it does nothing.
//   phase:  'off' -> 'aim' (pointing at a part, inspecting) -> 'grab' (pinched, part follows the hand)
//   action: one-shot edge for sound/HUD — null | 'grab' | 'remove' | 'drop'
// Once grabbed, the grab SURVIVES pose flicker as the hand moves (the old "it won't move" bug): only a clear
// 'open' hand or a lost hand (`present` false) releases it, and the caller further debounces 'open' over a
// few frames so a mid-drag wobble (the pinch loosening as you move) never drops or bins the part. A stray
// 'fist'/'point'/dropped reading keeps carrying. `id` = the reachable part (null = none in reach); `present`
// = a single controlling hand is here; `inBin` = the grabbed part is currently over the dustbin.
export const AIM_OFF = { phase: 'off', id: null, action: null };
export function aimStep(st, { pose, id, present = pose != null, inBin = false }) {
  if (st.phase === 'grab') {                                        // stuck to the hand; the pinch holds it
    if (pose === 'open' && inBin) return { ...AIM_OFF, id: st.id, action: 'remove' }; // opened over the bin: remove
    if (pose === 'open' || !present) return { ...AIM_OFF, id: st.id, action: 'drop' };  // opened / hand gone: snap home
    return { ...st, action: null };                                 // pinch (or a pose flicker) keeps carrying it
  }
  if (pose === 'pinch' && id != null) return { phase: 'grab', id, action: 'grab' };   // pinch a nearby part: grab it now
  if (pose === 'point' && id != null) return { phase: 'aim', id, action: null };       // POINT one finger at a part: inspect it (caller times the dwell before revealing the card)
  return AIM_OFF;                                                   // nothing pointing at / pinching a part: nothing armed
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

// Thumb sticking out (👍) vs tucked into a fist (✊): thumb tip far from the palm centroid,
// palm-normalized so it's scale- and rotation-robust. Only used to split 'thumbsup' from 'fist'
// (both have all four fingers curled) — a tucked fist-thumb lands ~0.7 palms out, a raised thumb ~1.5+.
export function thumbOut(hand) {
  return dist2d(hand[THUMB_TIP], handCenter(hand)) / palmSize(hand) > 0.8; // ponytail: 0.8 splits tucked/raised thumb; user's 👍 misread as fist at 0.9 — tune on real hands
}

// Classify the hand into the poses we act on: 'fist' | 'thumbsup' | 'pinch' | 'point' | 'open'.
// fist = every finger curled with the thumb tucked (grab & hold a part); thumbsup = fingers curled
// but the thumb raised (reset everything); pinch = a DELIBERATE thumb-on-index pinch (grab a part); point =
// only the index out (inspect a part); else open (neutral / two-palm zoom). The curled-finger cases are
// tested first, and pinch uses isPinchPose (not raw pinchStrength) so a relaxed palm or a fist — thumb near
// every fingertip — no longer misreads as a pinch (the over-sensitive grab you hit).
export function handPose(hand) {
  const idx = fingerExtended(hand, 8, 6);
  const mid = fingerExtended(hand, 12, 10);
  const rng = fingerExtended(hand, 16, 14);
  const pky = fingerExtended(hand, 20, 18);
  if (!idx && !mid && !rng && !pky) return thumbOut(hand) ? 'thumbsup' : 'fist';
  if (isPinchPose(hand)) return 'pinch';
  return (idx && !mid && !rng && !pky) ? 'point' : 'open';
}

// Parse a speech transcript into an axis-lock command: 'x' | 'y' | 'z' to freeze that rotation axis,
// 'unlock' to free all, null for anything else. Pure so it's unit-tested here, not against a live mic.
// Word-boundary matched so "lock the x axis" hits 'x' but "explode" or "extra" never do; "unlock" wins
// first (it contains "lock" too). Homophones the recognizer emits for single letters ('why'→y, 'zee'→z,
// 'ex'/'axe'→x) are folded in so a spoken letter that gets transcribed as a word still lands.
const AXIS_WORD = { x: ['x', 'ex', 'axe'], y: ['y', 'why'], z: ['z', 'zee', 'zed'] };
export function axisFromVoice(t) {
  if (/\b(unlock|free|release)\b/.test(t)) return 'unlock';
  if (!t.includes('lock')) return null;
  for (const a in AXIS_WORD) if (AXIS_WORD[a].some((w) => new RegExp(`\\b${w}\\b`).test(t))) return a;
  return null;
}

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

