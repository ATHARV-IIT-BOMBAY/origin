// gestures.test.js — the one runnable check for the gesture math.  Run: node gestures.test.js
import assert from 'node:assert';
import { palmSize, handCenter, pinchStrength, isPinching, twoHandSpread, landmarkToWorld } from './gestures.js';

// Build a synthetic 21-landmark hand: wrist at (0.5,0.9), middle knuckle at (0.5,0.6)
// => palmSize = 0.3. Thumb/index tips are passed in so we can force open vs pinched.
function makeHand({ thumb, index }) {
  const h = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.6, z: 0 }));
  h[0]  = { x: 0.50, y: 0.90, z: 0 }; // wrist
  h[5]  = { x: 0.45, y: 0.60, z: 0 }; // index base
  h[9]  = { x: 0.50, y: 0.60, z: 0 }; // middle base
  h[13] = { x: 0.55, y: 0.60, z: 0 }; // ring base
  h[17] = { x: 0.60, y: 0.60, z: 0 }; // pinky base
  h[4]  = thumb;
  h[8]  = index;
  return h;
}
const shiftX = (hand, dx) => hand.map(p => ({ x: p.x + dx, y: p.y, z: p.z }));

const open    = makeHand({ thumb: { x: 0.35, y: 0.40, z: 0 }, index: { x: 0.65, y: 0.40, z: 0 } }); // tips 0.30 apart
const pinched = makeHand({ thumb: { x: 0.49, y: 0.45, z: 0 }, index: { x: 0.51, y: 0.45, z: 0 } }); // tips ~0.02 apart

assert.ok(Math.abs(palmSize(open) - 0.3) < 1e-9, 'palmSize should be 0.3');
assert.ok(pinchStrength(open) < 0.2, `open hand should read un-pinched, got ${pinchStrength(open)}`);
assert.ok(pinchStrength(pinched) > 0.9, `closed hand should read pinched, got ${pinchStrength(pinched)}`);
assert.ok(!isPinching(open) && isPinching(pinched), 'isPinching must flip between the two');

assert.ok(Math.abs(handCenter(open).x - 0.5) < 0.05, 'palm center x should sit near 0.5');

const near = twoHandSpread(open, shiftX(open, 0.2));
const far  = twoHandSpread(open, shiftX(open, 0.5));
assert.ok(far > near, `spread must grow as hands separate (${near} -> ${far})`);

// landmarkToWorld: image center maps to scene origin (at the fixed hand plane), and X is
// mirrored so a landmark on the image's right lands on the scene's left.
const mid = landmarkToWorld({ x: 0.5, y: 0.5, z: 0 }, 4, 1.5);
assert.ok(Math.abs(mid.x) < 1e-9 && Math.abs(mid.y) < 1e-9, 'image center -> world origin (x,y)');
assert.ok(landmarkToWorld({ x: 0.9, y: 0.5, z: 0 }, 4).x < 0, 'X is mirrored');
assert.ok(landmarkToWorld({ x: 0.5, y: 0.9, z: 0 }, 4).y < 0, 'image bottom -> lower Y');

console.log('gestures.test.js: all assertions passed ✓');
