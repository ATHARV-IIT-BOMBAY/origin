// gestures.test.js — the one runnable check for the gesture math.  Run: node gestures.test.js
import assert from 'node:assert';
import { palmSize, handCenter, pinchStrength, isPinching, isPinchPose, twoHandSpread, twoHandAngle, rollDelta, landmarkToWorld, handPose, fitTransform, aimStep, AIM_OFF, palmPlane, pinchPoint, axisFromVoice } from './gestures.js';

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

// isPinchPose is the STRICT gate handPose uses to fire a grab — it must reject the two things the user saw
// misread as a pinch: a flat open palm, and a fist (thumb tucked near EVERY fingertip). A real pinch closes
// the thumb on the index SPECIFICALLY, so it's the only one where the index is distinctly the nearest tip.
assert.ok(isPinchPose(pinched), 'thumb+index touching, middle away => a real pinch');
assert.ok(!isPinchPose(open), 'an open palm (tips wide apart) is not a pinch');
// a fist: thumb tucked in the palm, as near the middle tip (0.5,0.6) as the index — close enough to fool the
// absolute-distance test, but the relative guard (index must be the closest) rejects it. This is the misread.
const fistGrip = makeHand({ thumb: { x: 0.50, y: 0.58, z: 0 }, index: { x: 0.52, y: 0.60, z: 0 } });
assert.ok(!isPinchPose(fistGrip), 'a fist (thumb near every fingertip, not the index specifically) is not a pinch');

assert.ok(Math.abs(handCenter(open).x - 0.5) < 0.05, 'palm center x should sit near 0.5');

// pinchPoint sits at the thumb/index midpoint — up near the fingertips, NOT back at the palm centroid
// (0.66 here). That offset is the whole point: you pinch AT a part with your fingers, not your wrist.
const pp = pinchPoint(pinched);
assert.ok(Math.abs(pp.x - 0.5) < 1e-9 && Math.abs(pp.y - 0.45) < 1e-9, 'pinchPoint is the thumb/index midpoint');
assert.ok(pp.y < handCenter(pinched).y - 0.1, 'pinch point sits well toward the fingertips, not the palm centroid');

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
// thumbs-up = fingers curled (like a fist) but the thumb raised clear of the palm => the reset gesture.
const thumbsup = poseHand({ index: false, middle: false, ring: false, pinky: false });
thumbsup[4] = { x: 0.50, y: 0.15, z: 0 }; // thumb tip straight up, far from the palm centroid
assert.equal(handPose(thumbsup), 'thumbsup', 'fingers curled + thumb raised => thumbs-up (reset)');

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

// aimStep: PROXIMITY + POSE model. The caller passes `id` = the part nearest the hand on screen when it's
// within reach (null = nothing close). POINTING one finger at a part INSPECTS it (the caller times the dwell);
// a PINCH while a part is in reach GRABs it DIRECTLY (no point/fist first — that was the grab-never-fires bug);
// OPEN releases (over the bin => removed, else snaps home). A FIST does NOT grab here (it rotates the whole
// model, in main.js) and must not inspect; an OPEN hand near a part is neutral and does not inspect either.
// The grab must survive the tracker flickering the pose mid-carry; a pinch carried through the bin must NOT delete.
{
  const step = (st, i) => aimStep(st, i);

  // nothing in reach arms nothing — a fist, an open hand, whatever, with no part nearby
  assert.equal(step(AIM_OFF, { pose: 'pinch', id: null }).phase, 'off', 'a pinch with no part in reach grabs nothing');
  assert.equal(step(AIM_OFF, { pose: 'open', id: null }).phase, 'off', 'an open hand over empty space arms nothing');

  // an OPEN hand near a part is neutral now — it does NOT inspect (only a deliberate point does)
  assert.equal(step(AIM_OFF, { pose: 'open', id: 3 }).phase, 'off', 'an open hand near a part is neutral, not an inspect');

  // POINTING one finger at a part within reach => inspecting it (info card, after the caller's dwell)
  let s = step(AIM_OFF, { pose: 'point', id: 3 });
  assert.ok(s.phase === 'aim' && s.id === 3 && s.action === null, 'pointing at a part within reach inspects it');

  // a FIST near a part must NOT grab and must NOT inspect — a fist is whole-model rotate, not a grab
  assert.equal(step(AIM_OFF, { pose: 'fist', id: 3 }).phase, 'off', 'a fist near a part neither grabs nor inspects (it rotates the model)');

  // PINCH while a part is in reach => grab it directly, no point/fist first (the whole fix)
  s = step(AIM_OFF, { pose: 'pinch', id: 3 });
  assert.ok(s.phase === 'grab' && s.action === 'grab' && s.id === 3, 'a pinch near a part grabs it at once, firing once');

  // dragging: the grab MUST survive the tracker flickering the pose/target as the hand moves — the old bug.
  const held = step(s, { pose: 'pinch', id: 3 });
  assert.ok(held.phase === 'grab' && held.action === null, 'holding the pinch keeps carrying — no repeat action');
  assert.equal(step(held, { pose: 'point', id: 9, present: true }).phase, 'grab', 'pose/target flickering mid-carry does NOT drop it');
  assert.equal(step(held, { pose: null,    id: null, present: true }).phase, 'grab', 'a dropped pose reading (hand still there) does NOT drop it');
  assert.equal(step(held, { pose: 'fist', id: null, present: true }).phase, 'grab', "a flicker to 'fist' (rotate) mid-carry does NOT drop it either");
  assert.equal(step(held, { pose: null, id: null, present: true }).id, 3, 'the carried part stays locked even as the nearest id changes');

  // removal is ONLY the deliberate open-over-bin: opening the hand over the bin removes it, firing once
  const binned = step(held, { pose: 'open', id: null, present: true, inBin: true });
  assert.ok(binned.action === 'remove' && binned.id === 3 && binned.phase === 'off', 'opening the hand over the bin removes it');

  // ...but carrying a fist THROUGH the bin without opening must NOT delete it (kills the proximity/timer bug)
  assert.equal(step(held, { pose: 'pinch', id: 3, present: true, inBin: true }).phase, 'grab', 'a pinch carried over the bin keeps holding — no auto-delete');

  // opening away from the bin snaps it home; losing the hand also drops home (a lost hand never removes)
  assert.equal(step(held, { pose: 'open', id: null, present: true, inBin: false }).action, 'drop', 'opening away from the bin drops it home');
  assert.equal(step(held, { pose: null,   id: null, present: false }).action, 'drop', 'losing the controlling hand drops it home, not removed');
  assert.ok(step(held, { pose: null, id: null, present: false }).phase === 'off', 'and the machine resets');

  // while inspecting: pointing nearer another part switches the target; pointing away from all parts abandons it
  assert.equal(step(step(AIM_OFF, { pose: 'point', id: 3 }), { pose: 'point', id: 7 }).id, 7, 'the nearer part switches inspection');
  assert.equal(step(step(AIM_OFF, { pose: 'point', id: 3 }), { pose: 'point', id: null }).phase, 'off', 'pointing away from all parts abandons inspection');

  // a thumbs-up near a part must NOT inspect or grab — it's the reset, resolved elsewhere
  assert.equal(step(AIM_OFF, { pose: 'thumbsup', id: 3 }).phase, 'off', 'thumbs-up near a part is not an inspect/grab (reset is separate)');
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

// axisFromVoice — voice command parsing (drives the mic-only axis lock)
{
  assert.equal(axisFromVoice('lock x'), 'x', 'lock x freezes the X axis');
  assert.equal(axisFromVoice('lock the y axis'), 'y', 'natural phrasing still resolves the axis');
  assert.equal(axisFromVoice('lock z please'), 'z');
  assert.equal(axisFromVoice('why'), null, 'a bare axis word without "lock" does nothing');
  assert.equal(axisFromVoice('unlock'), 'unlock', 'unlock frees all axes');
  assert.equal(axisFromVoice('unlock the x axis'), 'unlock', 'unlock wins even though it contains "lock" and "x"');
  assert.equal(axisFromVoice('explode'), null, 'an unrelated command is not a lock');
  assert.equal(axisFromVoice('lock'), null, '"lock" with no axis is ambiguous → no-op, not a wrong axis');
  assert.equal(axisFromVoice('lock ex'), 'x', 'recognizer homophone "ex" maps to X');
}

console.log('gestures.test.js: all assertions passed ✓');
