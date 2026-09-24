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
