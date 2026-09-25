// gestures.test.js — the one runnable check for the gesture math.  Run: node gestures.test.js
import assert from 'node:assert';
import { palmSize, handCenter, pinchStrength, isPinching, twoHandSpread, twoHandAngle, rollDelta, landmarkToWorld, handPose, fitTransform, aimStep, AIM_OFF, palmPlane } from './gestures.js';

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

// twoHandAngle: a horizontal pair reads ~0; lifting the second hand straight up rotates the
// joining line toward -90° (image Y grows downward, so "up" is a smaller y => negative angle).
assert.ok(Math.abs(twoHandAngle(open, shiftX(open, 0.3))) < 1e-9, 'level hands => angle ~0');
const lifted = open.map(p => ({ x: p.x + 0.3, y: p.y - 0.3, z: p.z })); // second hand up and to the right
assert.ok(twoHandAngle(open, lifted) < 0, 'lifting the far hand tips the roll angle negative');

// rollDelta: zooming (both pinch) => no roll no matter the twist; jitter below the deadzone => no
// roll; a deliberate twist beyond it => roll, sign-flipped by mirror. First frame (null) => 0.
assert.equal(rollDelta(null, 0.5, {}), 0, 'no previous angle => no roll');
assert.equal(rollDelta(0.0, 0.5, { zooming: true }), 0, 'zooming suppresses roll entirely');
assert.equal(rollDelta(0.0, 0.005, { deadzone: 0.012 }), 0, 'sub-deadzone wobble => no roll');
assert.ok(rollDelta(0.0, 0.2, { deadzone: 0.012, mirror: -1 }) < 0, 'a real twist rolls, mirrored to negative');
assert.ok(Math.abs(rollDelta(0.0, 0.2, { mirror: 1 }) - 0.2) < 1e-9, 'unmirrored roll equals the raw angle change');
// ...and the upper guard: if the tracker swaps which hand is first, the joining line flips ~180°.
// That must NOT roll the model, while a fast-but-human twist still must.
assert.equal(rollDelta(0.2, 0.2 - Math.PI, {}), 0, 'a ~180° flip (hands swapped in the tracker list) is rejected');
assert.equal(rollDelta(2.9, -2.9, {}), 0, 'the same flip across the ±π seam is rejected too');
assert.ok(Math.abs(rollDelta(0.0, 0.3, { mirror: 1 }) - 0.3) < 1e-9, 'a fast but human twist still rolls');

// landmarkToWorld: image center maps to scene origin (at the fixed hand plane), and X is
// mirrored so a landmark on the image's right lands on the scene's left.
const mid = landmarkToWorld({ x: 0.5, y: 0.5, z: 0 }, 4, 1.5);
assert.ok(Math.abs(mid.x) < 1e-9 && Math.abs(mid.y) < 1e-9, 'image center -> world origin (x,y)');
assert.ok(landmarkToWorld({ x: 0.9, y: 0.5, z: 0 }, 4).x < 0, 'X is mirrored');
assert.ok(landmarkToWorld({ x: 0.5, y: 0.9, z: 0 }, 4).y < 0, 'image bottom -> lower Y');

// handPose: build a hand where each finger is independently extended or curled. Extended
// finger => tip far from wrist (small y); curled => tip pulled back near the palm (large y).
// Thumb tip sits off to the side so the hand never reads as a pinch here.
function poseHand({ index, middle, ring, pinky }) {
  const h = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.6, z: 0 }));
  h[0] = { x: 0.50, y: 0.90, z: 0 };            // wrist
  h[9] = { x: 0.50, y: 0.60, z: 0 };            // middle knuckle => palmSize 0.3
  h[4] = { x: 0.30, y: 0.60, z: 0 };            // thumb tip, far from index => not a pinch
  const finger = (mcpX, extended, tip, pip) => {
    h[pip] = { x: mcpX, y: extended ? 0.45 : 0.55, z: 0 };
    h[tip] = { x: mcpX, y: extended ? 0.20 : 0.72, z: 0 };
  };
  finger(0.45, index,  8,  6);
  finger(0.50, middle, 12, 10);
  finger(0.55, ring,   16, 14);
  finger(0.60, pinky,  20, 18);
  return h;
}
const pointing = poseHand({ index: true,  middle: false, ring: false, pinky: false });
const flat     = poseHand({ index: true,  middle: true,  ring: true,  pinky: true });
const fist     = poseHand({ index: false, middle: false, ring: false, pinky: false });
assert.equal(handPose(pointing), 'point', 'index-only extended => point');
assert.equal(handPose(flat), 'open', 'all fingers extended => open');
assert.equal(handPose(fist), 'fist', 'all fingers curled => fist (parks/ignores the hand)');
assert.equal(handPose(pinched), 'pinch', 'thumb+index together => pinch');

// fitTransform: a model whose geometry sits FAR from its own origin (min 10..14) must still land
// centered on the stage with its feet on the floor — the bug was that scaling after centering
// flung it back off-screen. Apply the returned transform and check the final world box.
{
  const min = { x: 10, y: 10, z: 10 }, max = { x: 12, y: 14, z: 12 }; // 2×4×2, tallest side 4
  const t = fitTransform(min, max, 1.8, -1.6);
  assert.ok(Math.abs(t.scale - 1.8 / 4) < 1e-9, 'scales tallest side (4) to target 1.8');
  const world = (p, c) => t.position[c] + t.scale * p[c];             // world = position + scale*local
  assert.ok(Math.abs((world(min, 'x') + world(max, 'x')) / 2) < 1e-9, 'centered on X after scaling');
  assert.ok(Math.abs((world(min, 'z') + world(max, 'z')) / 2) < 1e-9, 'centered on Z after scaling');
  assert.ok(Math.abs(world(min, 'y') - (-1.6)) < 1e-9, 'feet rest exactly on the floor');
}

// aimStep: the pinch-free part-removal machine, driven by a fake clock. The point of the long dwell is
// that nothing gets grabbed by accident; the point of the rewrite is that no pinch is involved anywhere,
// so the finger never has to change pose to grab or drag (forming the pinch was what the tracker kept
// misreading). A fling (fast) or a pull-clear-then-release removes; a release near home just drops it.
{
  const O = { dwellMs: 600 };
  const step = (st, i) => aimStep(st, i, O);

  // only a steady point at a real part may start the machine — a pinch is the rotate gesture, hands off
  assert.equal(step(AIM_OFF, { pose: 'pinch', id: 3, now: 0 }).phase, 'off', 'a pinch never arms removal (it rotates)');
  assert.equal(step(AIM_OFF, { pose: 'open',  id: 3, now: 0 }).phase, 'off', 'an open hand arms nothing');
  assert.equal(step(AIM_OFF, { pose: 'point', id: null, now: 0 }).phase, 'off', 'pointing at empty space arms nothing');

  // point at part 3 and hold: the dwell fills, then the part grabs onto the finger — no pinch needed
  let s = step(AIM_OFF, { pose: 'point', id: 3, now: 1000 });
  assert.equal(s.phase, 'aim', 'pointing at a part starts the dwell');
  s = step(s, { pose: 'point', id: 3, now: 1300 });
  assert.ok(s.phase === 'aim' && Math.abs(s.progress - 0.5) < 1e-9, 'half-way through the dwell, still only aiming');
  s = step(s, { pose: 'point', id: 3, now: 1600 });
  assert.ok(s.phase === 'grab' && s.action === 'grab' && s.id === 3, 'a full dwell grabs the part onto the finger, firing once');

  // dragging: keep pointing and it stays grabbed (following the finger); a slow pull-out must NOT remove
  const held = step(s, { pose: 'point', id: 3, now: 1700 });
  assert.ok(held.phase === 'grab' && held.action === null, 'holding the point keeps dragging it — no repeat action');
  assert.equal(step(held, { pose: 'point', id: 3, now: 2000, pulled: true }).phase, 'grab', 'a slow pull-out only repositions — it does not auto-remove');

  // removal, three ways in: a fast fling, pulling it clear then letting go, or yanking off-frame while clear
  assert.equal(step(held, { pose: 'point', id: 3, now: 1800, fling: true }).action, 'remove', 'a fast fling throws it off even mid-point');
  const out = step(held, { pose: 'open', id: null, now: 2000, pulled: true });
  assert.ok(out.action === 'remove' && out.id === 3 && out.phase === 'off', 'releasing a pulled-clear part removes it');
  assert.equal(step(held, { pose: null, id: null, now: 2000, pulled: true }).action, 'remove', 'yanking the hand off-frame while clear still removes');

  // ...but a release short of the eject distance just snaps it home
  assert.equal(step(held, { pose: 'open', id: null, now: 2000, pulled: false }).action, 'drop', 'letting go near home drops it back, not removed');

  // escape hatches during the dwell: aim elsewhere restarts it, dropping the point abandons it
  assert.ok(step(step(AIM_OFF, { pose: 'point', id: 3, now: 0 }), { pose: 'point', id: 7, now: 100 }).id === 7, 'aiming at a different part restarts the dwell there');
  assert.equal(step(step(AIM_OFF, { pose: 'point', id: 3, now: 0 }), { pose: 'open', id: null, now: 100 }).phase, 'off', 'dropping the point abandons the dwell');
}

// palmPlane: the cut plane the section gesture hands to the renderer. A hand held flat against the
// image plane faces the camera, so its normal must be +Z; rolling the pinky side away from the
// camera must tip that normal sideways, and the normal must always be unit length or the clipping
// plane's distance test is meaningless.
{
  const flatPalm = makeHand({ thumb: { x: 0.3, y: 0.6, z: 0 }, index: { x: 0.45, y: 0.2, z: 0 } });
  const p = palmPlane(flatPalm, 4, 1.5);
  assert.ok(Math.hypot(p.normal.x, p.normal.y, p.normal.z) - 1 < 1e-9, 'normal is unit length');
  assert.ok(p.normal.z > 0.999, 'a hand flat to the camera gives a +Z normal (palm faces you)');

  const rolled = flatPalm.map((q, i) => i === 17 ? { ...q, z: 0.5 } : q); // pinky knuckle pushed away
  const r = palmPlane(rolled, 4, 1.5);
  assert.ok(r.normal.x < -0.5, 'rolling the pinky side away tips the normal off-axis');
  assert.ok(Math.abs(Math.hypot(r.normal.x, r.normal.y, r.normal.z) - 1) < 1e-9, 'still unit length when tilted');

  // three collinear palm points have no plane; returning NaN would clip the whole model away
  const flat = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  assert.equal(palmPlane(flat).normal.z, 1, 'a degenerate palm falls back to +Z instead of NaN');
}

console.log('gestures.test.js: all assertions passed ✓');
