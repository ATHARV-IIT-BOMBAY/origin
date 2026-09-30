// Origin — rotate & zoom a 3D model with webcam hand gestures.
// Flow: webcam -> MediaPipe HandLandmarker -> gesture math (gestures.js) -> three.js transform.
// No LLM, no backend. The hand model is only a *sensor*; the interaction is hand-written geometry.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { HandLandmarker, FilesetResolver, DrawingUtils }
  from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18';
import { handCenter, pinchStrength, twoHandSpread, twoHandAngle, rollDelta, landmarkToWorld, handPose, fitTransform, aimStep, AIM_OFF, palmPlane, pinchPoint, INDEX_TIP, axisFromVoice } from './gestures.js';

// ---- Tuning knobs. A webcam is a messy sensor; these are the calibration dials. ----
// The first four are `let` because the on-screen calibration panel adjusts them live.
let ROT_SPEED = 6.0;      // how far a hand move rotates the model
let ZOOM_SPEED = 6.0;     // how strongly two-hand spread scales the model
let SMOOTH = 0.20;        // 0..1 low-pass ease; lower = smoother but laggier
const GRAB_SMOOTH = 0.5;  // snappier ease for a carried part so it sticks to the fingertips (still filters MediaPipe jitter)
let MIRROR_X = -1;        // flip so moving your hand right rotates the model right
const SCALE_MIN = 0.3, SCALE_MAX = 4.0;
const HAND_SPAN = 4.2;    // how wide the tracked hand maps into the 3D scene
const HAND_DEPTH = 1.5;   // how strongly landmark depth pushes hand joints in/out
const IDLE_SPIN = 0.0015; // lazy auto-rotate (rad/frame) when you're not controlling it
// Flick-to-spin: while pinch-dragging we track the swipe velocity; on release the model keeps
// spinning and eases to rest (friction). CAPTURE = how much the latest frame feeds the estimate.
const SPIN_FRICTION = 0.96, SPIN_CAPTURE = 0.5, SPIN_MIN = 0.001;
const EXPLODE_K = 1.6;                        // full-explosion expansion: parts push out from center
const EXPLODE_SPEED = 3.5;                    // ponytail: explode is INCREMENTAL — widening the two-fist gap ADDS, narrowing removes, releasing HOLDS it (like zoom), so an exploded model persists while you then zoom/rotate. Tune on real hands.
const TWIST_DEADZONE = 0.012;                 // rad/frame of two-hand twist to ignore as jitter, so a zoom/hold doesn't drift into roll
const BASE_EMISSIVE = 0.6, HIGHLIGHT_EMISSIVE = 2.4; // part glow: resting vs. aimed-at
// Part extraction, proximity + pinch, timer-free. Bring your hand NEAR a part (screen-space, within
// REACH) to INSPECT it (info card, blue glow, no commitment); PINCH to GRAB it (turns red) — it then
// follows your hand 1:1 in the model's own space, and the grab survives the tracker's pose flickering
// as you move (that flicker dropping the grab was the "it won't move" bug). OPEN your hand to let go:
// over the DUSTBIN it's removed, anywhere else it snaps home. No timer, so nothing deletes by dwell and
// nothing deletes just for being carried near the bin — removal needs the deliberate open-over-bin. A
// FIST rotates the whole model (not a grab); an OPEN hand in empty space is neutral. See gestures.js.
const HL_INFO = 0x2ad0ff, HL_GRAB = 0xff2a2a;  // emissive tint: blue while inspecting, red once it's grabbed and moveable
const BIN_HIT = 0.44;               // ponytail: SCREEN-space (NDC, aspect-corrected) radius that counts a carried part as "over the bin" — depth-independent, so MediaPipe's noisy/compressed hand-z can't block a delete. Anchored to the bin's MOUTH (see grab block), and generous because removal still needs a deliberate open, not mere proximity. Tune on real hardware
const REACH = 0.34;                 // ponytail: screen-space (NDC, aspect-corrected) radius that counts a part as "in reach" of the hand; tune on real hardware
const RANK = { pinch: 4, fist: 3, point: 2, thumbsup: 1, open: 0 }; // when a 2nd hand is present but it's not a committed zoom/explode pair, act on the single highest-ranked (dominant) hand
const DWELL_MS = 1500;      // ponytail: hold a POINT on a part this long before the info card commits — a glance shouldn't fire it (user asked ~2s; 1.5s reads snappier, tune)
const RELEASE_FRAMES = 5;   // ponytail: consecutive clearly-open frames before a grabbed part releases, so the pinch loosening as you drag doesn't drop/bin it (~0.15s at 30fps)
const THUMB_HOLD_MS = 700;  // ponytail: hold 👍 this long before reset fires — a fist momentarily misread as thumbs-up must never wipe your work
const EJECT_SPEED = 0.055;  // dissolve speed per frame (~0.3s at 60fps)
const GHOST_FACTOR = 0.16;  // the rest of the model dims to this fraction of its opacity while a part is grabbed
const CUT_SMOOTH = 0.15;    // ease on the section plane — MediaPipe's landmark depth is noisy, and an unsmoothed plane strobes
const CUT_PARKED = 1e6;     // plane constant that puts the cut so far away nothing is clipped (see clipPlane)
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm';

const $ = (id) => document.getElementById(id);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const video = $('video'), overlay = $('overlay'), stateEl = $('state'), errEl = $('err'), poseEl = $('pose');
const hud = $('hud');
// cockpit mode: recede the setup controls when a hand is driving AND the mouse is idle (see loop()).
let lastHandTs = 0, lastMouseTs = 0;
addEventListener('pointermove', () => { lastMouseTs = performance.now(); }, { passive: true });
addEventListener('pointerdown', () => { lastMouseTs = performance.now(); }, { passive: true });
const modeEl = $('modePill'), recEl = $('rec');

// ---------- HUD: live mode pill + stats readout ----------
const MODE_LABEL = { rotate:'ROTATE', zoom:'ZOOM', explode:'EXPLODE', inspect:'INSPECT', lock:'LOCKED', grab:'EXTRACT', section:'SECTION', park:'PARKED', pause:'PAUSED', idle:'IDLE' };
function updateModePill(m) { if (modeEl) { modeEl.textContent = MODE_LABEL[m] || m.toUpperCase(); modeEl.className = 'pill ' + m; } }
let lastHands = 0;
function updateStats(nHands = lastHands) {
  lastHands = nHands;
  $('stHands').textContent = nHands;
  $('stParts').textContent = removed.length ? `${parts.length - removed.length}/${parts.length}` : parts.length;
  $('stScale').textContent = Math.round(current.scale * 100) + '%';
  $('stExpl').textContent = Math.round(current.explode * 100) + '%';
  $('stGone').textContent = removed.length;
}

// ---------- Web Audio: tiny synth blips for gesture feedback (no audio files) ----------
// A gesture is invisible until it "clicks" — a short tone on each mode change makes the
// interface feel physical. Ctx is created on a user click (autoplay policy) and reused.
let audioCtx = null;
function initAudio() {
  if (!audioCtx) { try { audioCtx = new (window.AudioContext || window.webkitAudioContext)(); } catch {} }
  if (audioCtx && audioCtx.state === 'suspended') audioCtx.resume();
}
function blip(freq = 660, dur = 0.08, type = 'sine', peak = 0.06) {
  if (!audioCtx) return;
  const t = audioCtx.currentTime, osc = audioCtx.createOscillator(), g = audioCtx.createGain();
  osc.type = type; osc.frequency.setValueAtTime(freq, t);
  g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(peak, t + 0.01);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  osc.connect(g).connect(audioCtx.destination); osc.start(t); osc.stop(t + dur + 0.02);
}
const SFX = {                       // sound played when we ENTER each mode (fires once per transition)
  rotate:  () => blip(520, 0.07, 'sine'),
  zoom:    () => blip(760, 0.09, 'triangle'),
  explode: () => blip(300, 0.12, 'sawtooth', 0.05),
  inspect: () => blip(880, 0.05, 'square', 0.03),
  grab:    () => { blip(1180, 0.06, 'square', 0.04); setTimeout(() => blip(1480, 0.10, 'square', 0.05), 70); }, // two-tone: part grabbed onto the finger
  section: () => { blip(420, 0.05, 'square', 0.03); setTimeout(() => blip(1050, 0.16, 'sine', 0.04), 50); }, // a "slice"
  park:    () => blip(200, 0.10, 'sine'),
  pause:   () => blip(150, 0.16, 'sine'),
};
let lastMode = '';
function setMode(m) { if (m === lastMode) return; lastMode = m; updateModePill(m); SFX[m]?.(); }

// ---------- three.js scene ----------
// preserveDrawingBuffer lets the Snapshot button read the canvas back as a PNG.
const renderer = new THREE.WebGLRenderer({ canvas: $('three'), antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
// Filmic tone mapping: without it the renderer clips anything brighter than 1.0 straight to flat white,
// so an uploaded model's glowing bits (Iron Man's arc reactor, faceplate, shiny metal under the key light)
// blow out into white blobs. ACES rolls highlights off to a bright colour instead. OutputPass (added below)
// is what actually applies this to the composed frame. ponytail: exposure is the one knob to turn if a whole
// class of models reads too hot/dim — wire it to a slider only if fixed 1.0 proves not enough.
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.9;
renderer.localClippingEnabled = true;   // per-material clipping, so the section plane cuts the model but not its own indicator
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0f);
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
camera.position.set(0, 0, 4);

const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture; // soft PBR reflections
scene.add(new THREE.HemisphereLight(0xffffff, 0x223344, 0.6));
const key = new THREE.DirectionalLight(0xffffff, 1.2); key.position.set(3, 4, 5); scene.add(key);  // was 2.0: a hot key blew the specular on shiny uploaded models to white

// Post-processing: bloom makes the emissive wireframe model and the hand joints glow,
// which is what turns a plain mesh into a "hologram". Threshold keeps merely-lit surfaces un-bloomed
// (0.82 sits above lit metal but below the emissive wireframe/highlights), so a real uploaded model
// no longer haloes its whole body — only the parts that genuinely emit light bloom.
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 0.7, 0.4, 0.82);
composer.addPass(bloom);
composer.addPass(new OutputPass());

// Faint "holo-deck" grid so the model reads as floating in a space, not on a black void.
const FLOOR_Y = -1.6;                 // baseline stage floor; the Ground slider moves the base from here
const grid = new THREE.GridHelper(20, 40, 0x1e6fff, 0x0a2a4a);
grid.position.y = FLOOR_Y; scene.add(grid);

// Holo-projector base: two counter-rotating tech rings under the model (bloom makes them glow).
const reticle = new THREE.Group(); reticle.position.y = FLOOR_Y + 0.05; scene.add(reticle);
const _up = new THREE.Vector3(0, 1, 0);
for (const [rIn, rOut, dir] of [[1.70, 1.86, 1], [1.96, 2.02, -1]]) {
  const ring = new THREE.Mesh(new THREE.RingGeometry(rIn, rOut, 96),
    new THREE.MeshBasicMaterial({ color: 0x2aa0ff, transparent: true, opacity: 0.55, side: THREE.DoubleSide }));
  ring.rotation.x = -Math.PI / 2; ring.userData.dir = dir; reticle.add(ring);
}

const pivot = new THREE.Group(); // we rotate/scale this; the model lives inside it
scene.add(pivot);

// ---------- dustbin: drag a grabbed part into this to remove it ----------
// Lives in SCENE space, not inside pivot, so it stays put on the stage while the model spins. Hidden
// (faded out) until a part is grabbed, then it rises in beside the stage; drag the grabbed part over it
// on screen and the part dissolves. An open can (body + rim + base) in a red holo skin, with a
// wire overlay to match the model's look. Never intercepts the aim ray (all meshes raycast to nothing).
const bin = new THREE.Group(); bin.visible = false; scene.add(bin);
let binShow = false;   // target visibility; the loop eases the bin's opacity toward it
let binArmed = false;  // a carried part is over the bin RIGHT NOW — the loop flares the bin so you can SEE the drop will delete (the missing feedback that made it feel like nothing ever landed "inside")
{
  const skin = new THREE.MeshStandardMaterial({ color: 0x2a0808, emissive: 0xff3018, emissiveIntensity: 0.6,
    metalness: 0.3, roughness: 0.4, transparent: true, opacity: 0, side: THREE.DoubleSide });
  const wireMat = new THREE.MeshBasicMaterial({ color: 0xff7a60, wireframe: true, transparent: true, opacity: 0 });
  const body = new THREE.Mesh(new THREE.CylinderGeometry(0.44, 0.34, 0.82, 28, 1, true), skin);
  const rim  = new THREE.Mesh(new THREE.TorusGeometry(0.44, 0.045, 10, 28), skin); rim.rotation.x = Math.PI / 2; rim.position.y = 0.41;
  const base = new THREE.Mesh(new THREE.CircleGeometry(0.34, 28), skin); base.rotation.x = -Math.PI / 2; base.position.y = -0.41;
  const wire = new THREE.Mesh(new THREE.CylinderGeometry(0.44, 0.34, 0.82, 14, 3, true), wireMat);
  for (const m of [body, rim, base, wire]) { m.raycast = () => {}; bin.add(m); }
  bin.userData.mats = [skin, wireMat];
}
bin.position.set(1.0, FLOOR_Y + 0.35, 0.7);   // ponytail: front-right, pulled in from the old 1.7 so the bin sits well inside frame. The hit test is screen-space now (BIN_HIT), so its z no longer has to be reachable — but keeping it near the stage still reads clearly. Tune with BIN_HIT
function showBin(on) { binShow = on; }

// Move the whole stage (floor grid + projector reticle) to a new base height and lift the model with
// it, so the model's feet stay planted on the base wherever you set it. fitToView grounds to FLOOR_Y.
function setGround(y) { grid.position.y = y; reticle.position.y = y + 0.05; pivot.position.y = y - FLOOR_Y; }

// ---------- section plane: your palm is the knife ----------
// Three points on the back of the hand give a plane (gestures.palmPlane); three.js clips every
// fragment on its negative side, so the model is cut open live and turning your palm over swaps
// which half survives. The plane is handed to the MODEL's materials rather than to
// renderer.clippingPlanes, because a global plane would also slice the indicator quad drawn on it.
// It's never removed from the array — disarming parks it a million units away instead. Changing the
// NUMBER of clipping planes recompiles every shader, and doing that on a gesture would stutter.
// ponytail: an open cut, no cap — a sectioned solid reads hollow. Capping needs a stencil pass.
const clipPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), CUT_PARKED);
const CLIP = [clipPlane];                      // one shared array: mutate the plane, never the list
function applyClip(root) {
  root.traverse(o => { if (o.isMesh) for (const m of [o.material].flat()) if (m) m.clippingPlanes = CLIP; });
}
const cutViz = new THREE.Group(); cutViz.visible = false; scene.add(cutViz);
{
  const quad = new THREE.PlaneGeometry(4.4, 4.4);
  cutViz.add(
    new THREE.Mesh(quad, new THREE.MeshBasicMaterial({
      color: 0x35c8ff, transparent: true, opacity: 0.06, side: THREE.DoubleSide, depthWrite: false })),
    new THREE.LineSegments(new THREE.EdgesGeometry(quad), new THREE.LineBasicMaterial({
      color: 0x66e0ff, transparent: true, opacity: 0.75 })));
}
const _cutN = new THREE.Vector3(0, 0, 1), _cutP = new THREE.Vector3(), _pgZ = new THREE.Vector3(0, 0, 1);
const _tn = new THREE.Vector3(), _tp = new THREE.Vector3();   // per-frame scratch, kept out of the GC's way
let sectionOn = false;
function updateCut(hand) {          // ease the plane toward the palm — raw landmark depth strobes
  const { point, normal } = palmPlane(hand, HAND_SPAN, HAND_DEPTH);
  _cutN.lerp(_tn.set(normal.x, normal.y, normal.z), CUT_SMOOTH).normalize();
  _cutP.lerp(_tp.set(point.x, point.y, point.z), CUT_SMOOTH);
  clipPlane.setFromNormalAndCoplanarPoint(_cutN, _cutP);
  cutViz.position.copy(_cutP);
  cutViz.quaternion.setFromUnitVectors(_pgZ, _cutN);
}
function setSection(on) {
  sectionOn = on;
  cutViz.visible = on;
  if (!on) clipPlane.constant = CUT_PARKED;   // park it rather than drop it: see the note above
  else { aim = AIM_OFF; locked = null; grabTip = null; binShow = false; ghostOthers(null); clearHighlight(); }
  $('secBtn').classList.toggle('on', on);
  initAudio(); blip(on ? 880 : 300, 0.09, 'square', 0.04);
}

// Hologram-styled part: translucent lit core + bright wireframe overlay (the bit that blooms).
// Grouped so a whole part can be raycast, highlighted, and flown out as a single unit.
function holoPart(geo, name, label, color = 0x0a2a4a, emissive = 0x0aa0ff) {
  const core = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
    color, emissive, emissiveIntensity: BASE_EMISSIVE,
    metalness: 0.3, roughness: 0.35, transparent: true, opacity: 0.8 }));
  const wire = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
    color: 0x66e0ff, wireframe: true, transparent: true, opacity: 0.45 }));
  core.userData.holoNative = true; wire.userData.holoWire = true;  // holoNative => keeps its own blue fill in normal/holo (still X-rays); wire shows in holo only
  const g = new THREE.Group(); g.add(core, wire);
  g.name = name; g.userData = { label, core };
  return g;
}

// Default model: a multi-part rocket (nose, tanks, engine, fins). It's recognizable, and its
// separately-named parts are exactly what the point-to-inspect and explode gestures act on.
function makeRocket() {
  const R = new THREE.Group(), r = 0.42;
  const nose = holoPart(new THREE.ConeGeometry(r, 0.9, 32), 'nose', 'Payload Fairing', 0x143a5a, 0x22c0ff);
  nose.position.y = 1.15;
  const payload = holoPart(new THREE.CylinderGeometry(r * 0.4, r * 0.4, 0.36, 16), 'payload', 'Satellite Payload', 0x0e2e1e, 0x35ffa6);
  payload.position.y = 0.92;   // tucked inside the fairing — the explode reveals it, like a real diagram
  const tankA = holoPart(new THREE.CylinderGeometry(r, r, 0.9, 32), 'tankA', 'Fuel Tank');
  tankA.position.y = 0.35;
  const tankB = holoPart(new THREE.CylinderGeometry(r, r, 0.9, 32), 'tankB', 'Oxidizer Tank');
  tankB.position.y = -0.55;
  const engine = holoPart(new THREE.CylinderGeometry(r * 0.55, r, 0.5, 32), 'engine', 'Engine Bell', 0x3a1e08, 0xff7a1a);
  engine.position.y = -1.2;
  R.add(nose, payload, tankA, tankB, engine);
  for (let i = 0; i < 3; i++) {                        // three stabilizer fins around the base
    const fin = holoPart(new THREE.BoxGeometry(0.06, 0.5, 0.42), 'fin' + i, 'Stabilizer Fin');
    const a = (i / 3) * Math.PI * 2;
    fin.position.set(Math.cos(a) * (r + 0.16), -0.95, Math.sin(a) * (r + 0.16));
    fin.rotation.y = -a;
    R.add(fin);
  }
  return R;
}
// ---- More built-in demo models, so a visitor with no .glb of their own can still try every gesture.
// Each is hand-built from named holoParts exactly like the rocket — that's the whole point: the parts
// carry labels and fly apart into a diagram. Chosen from the "Demo models" row in the HUD. ------------
function makeEngine() {   // turbofan, stacked along Y so two-fist explode fans it into a cutaway
  const E = new THREE.Group(), r = 0.5;
  const add = (p, y) => { p.position.y = y; E.add(p); return p; };
  add(holoPart(new THREE.ConeGeometry(0.12, 0.3, 24), 'spinner', 'Spinner Cone', 0x143a5a, 0x22c0ff), 1.15);
  add(holoPart(new THREE.CylinderGeometry(r, r, 0.22, 32), 'fan', 'Fan Stage'), 0.92);
  add(holoPart(new THREE.CylinderGeometry(r * 0.7, r * 0.9, 0.5, 32), 'lpc', 'Compressor'), 0.5);
  add(holoPart(new THREE.CylinderGeometry(r * 0.6, r * 0.7, 0.4, 32), 'burner', 'Combustor', 0x3a1e08, 0xff7a1a), 0.05);
  add(holoPart(new THREE.CylinderGeometry(r * 0.75, r * 0.6, 0.45, 32), 'turbine', 'Turbine'), -0.42);
  add(holoPart(new THREE.CylinderGeometry(r * 0.95, r * 0.75, 0.4, 32), 'nozzle', 'Exhaust Nozzle', 0x3a1e08, 0xff7a1a), -0.85);
  add(holoPart(new THREE.CylinderGeometry(r * 1.15, r * 1.15, 1.4, 40, 1, true), 'nacelle', 'Nacelle'), 0.25);  // open cowl; sits at centre so the core flies out of it
  return E;
}
function makeSatellite() {
  const S = new THREE.Group();
  S.add(holoPart(new THREE.BoxGeometry(0.8, 0.9, 0.8), 'bus', 'Satellite Bus'));
  const dish = holoPart(new THREE.SphereGeometry(0.5, 24, 12, 0, Math.PI * 2, 0, Math.PI / 2.4), 'dish', 'Comms Dish', 0x143a5a, 0x35ffa6);
  dish.rotation.x = Math.PI; dish.position.set(0, 0.1, 0.7); S.add(dish);
  for (const [s, tag] of [[-1, 'Solar Array · Port'], [1, 'Solar Array · Stbd']]) {
    const panel = holoPart(new THREE.BoxGeometry(1.5, 0.02, 0.7), 'panel' + s, tag, 0x0e2e1e, 0x35ffa6);
    panel.position.set(s * 1.25, 0.1, 0); S.add(panel);
  }
  const ant = holoPart(new THREE.CylinderGeometry(0.02, 0.02, 0.8, 8), 'ant', 'High-Gain Antenna'); ant.position.y = 0.85; S.add(ant);
  const thr = holoPart(new THREE.ConeGeometry(0.16, 0.3, 16), 'thr', 'Ion Thruster', 0x3a1e08, 0xff7a1a); thr.rotation.x = Math.PI; thr.position.y = -0.6; S.add(thr);
  return S;
}
function makeMolecule() {   // water, H₂O — atoms are the parts; bonds ride each hydrogen, so an explode pulls the atoms off
  const M = new THREE.Group();
  M.add(holoPart(new THREE.SphereGeometry(0.5, 32, 24), 'O', 'Oxygen', 0x4a0d0d, 0xff5a4a));
  const ang = 104.5 * Math.PI / 180, L = 1.05;
  for (const s of [-1, 1]) {
    const a = s * ang / 2 - Math.PI / 2;   // splay the two H's below the O at the real bond angle
    const hx = Math.cos(a) * L, hy = Math.sin(a) * L;
    const H = holoPart(new THREE.SphereGeometry(0.3, 24, 16), 'H' + s, 'Hydrogen', 0x1a2a3a, 0xbfe8ff);
    H.position.set(hx, hy, 0);
    const bond = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, L, 12), new THREE.MeshStandardMaterial({
      color: 0x0a2a4a, emissive: 0x2aa0ff, emissiveIntensity: 0.5, metalness: 0.3, roughness: 0.4, transparent: true, opacity: 0.7 }));
    bond.position.set(-hx / 2, -hy / 2, 0); bond.rotation.z = a - Math.PI / 2;   // back toward O, cylinder axis along the O–H line
    H.add(bond);   // child of H (not O) so partRoot still sees M's 3 atoms as the assembly level
    M.add(H);
  }
  return M;
}
function makeDrone() {   // quadcopter — body + 4 arms + 4 rotors; explodes into an assembly diagram
  const D = new THREE.Group();
  D.add(holoPart(new THREE.BoxGeometry(0.7, 0.24, 0.7), 'body', 'Flight Controller'));
  const gimbal = holoPart(new THREE.SphereGeometry(0.16, 20, 14), 'gimbal', 'Camera Gimbal', 0x143a5a, 0x35ffa6);
  gimbal.position.y = -0.22; D.add(gimbal);
  for (let i = 0; i < 4; i++) {
    const a = (i / 4) * Math.PI * 2 + Math.PI / 4, R = 0.62;
    const arm = holoPart(new THREE.BoxGeometry(0.6, 0.06, 0.08), 'arm' + i, 'Motor Arm');
    arm.position.set(Math.cos(a) * R * 0.5, 0.02, Math.sin(a) * R * 0.5); arm.rotation.y = -a; D.add(arm);
    const rotor = holoPart(new THREE.CylinderGeometry(0.28, 0.28, 0.03, 24), 'rotor' + i, 'Rotor', 0x3a1e08, 0xff7a1a);
    rotor.position.set(Math.cos(a) * R, 0.08, Math.sin(a) * R); D.add(rotor);
  }
  return D;
}
// ---- 4D hyperspace ------------------------------------------------------------------------------
// MediaPipe only hands me 3D-ish landmarks; nothing out there understands 4D. So this is pure geometry
// I wrote: rotate a 4-polytope's vertices in two 4D planes, then PROJECT 4D→3D by perspective — a vertex
// nearer along the 4th axis (w) projects larger, which is exactly why a tesseract reads as a small cube
// nested inside a big one, turning inside-out. Same divide as a camera lens, just one dimension up.
function rot4(p, a, i, j) {            // rotate 4-vector p by angle a in the (i,j) plane, in place
  const c = Math.cos(a), s = Math.sin(a), pi = p[i], pj = p[j];
  p[i] = pi * c - pj * s; p[j] = pi * s + pj * c;
}
function make4D(verts4, edges) {
  const G = new THREE.Group();
  const N = verts4.length, proj = Array.from({ length: N }, () => new THREE.Vector3());
  // Colour each edge by where its ends sit along the 4th axis: the two cubes stay cyan, and the struts
  // that ONLY exist in 4D (ends on opposite sides of w) glow amber — Atharv's "path that connects them".
  const sgn = w => w > 1e-6 ? 1 : w < -1e-6 ? -1 : 0;
  const OUT = [.5, .91, 1], IN = [.16, .55, 1], LINK = [1, .58, .08];
  const eGeo = new THREE.BufferGeometry();
  eGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(edges.length * 6), 3));
  const ec = new Float32Array(edges.length * 6);
  edges.forEach(([a, b], e) => {
    const sa = sgn(verts4[a][3]), sb = sgn(verts4[b][3]);
    const c = sa === sb ? (sa < 0 ? IN : OUT) : LINK;   // same 4D-level = a cube edge; crosses w = a 4D strut
    for (let h = 0; h < 6; h++) ec[e * 6 + h] = c[h % 3];
  });
  eGeo.setAttribute('color', new THREE.BufferAttribute(ec, 3));
  const lines = new THREE.LineSegments(eGeo, new THREE.LineBasicMaterial(
    { vertexColors: true, transparent: true, opacity: .95, blending: THREE.AdditiveBlending, depthWrite: false }));
  const vGeo = new THREE.BufferGeometry();
  vGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(N * 3), 3));
  const vc = new Float32Array(N * 3);
  for (let k = 0; k < N; k++) { const c = sgn(verts4[k][3]) < 0 ? IN : OUT; vc[k*3]=c[0]; vc[k*3+1]=c[1]; vc[k*3+2]=c[2]; }
  vGeo.setAttribute('color', new THREE.BufferAttribute(vc, 3));
  const nodes = new THREE.Points(vGeo, new THREE.PointsMaterial(
    { vertexColors: true, size: .17, transparent: true, opacity: .95, blending: THREE.AdditiveBlending, depthWrite: false }));
  lines.frustumCulled = nodes.frustumCulled = false;   // it pulses past its first-frame bounds as it tumbles
  G.add(lines, nodes);
  const D = 2.6, REVEAL = 2200;   // D: 4D viewer distance (nesting depth). REVEAL: ms to grow from one cube into 4D
  let born = performance.now();
  G.userData.tick = () => {
    const now = performance.now(), P = eGeo.attributes.position.array, Vp = vGeo.attributes.position.array;
    let s = Math.min(1, (now - born) / REVEAL); s = s * s * (3 - 2 * s);   // ease the "extrude into the 4th dimension" reveal
    const tr = Math.max(0, (now - born - REVEAL) / 1000);                  // tumble clock — starts only once it's fully formed
    const aXW = tr * .30, aZW = tr * .20, aYW = s >= 1 ? current.explode * Math.PI : 0;
    for (let k = 0; k < N; k++) {
      const q = verts4[k], p = [q[0], q[1], q[2], q[3] * s];   // grow w from 0: one cube separates into nested cubes + struts
      rot4(p, aXW, 0, 3); rot4(p, aZW, 2, 3); rot4(p, aYW, 1, 3);
      const f = D / (D - p[3]);                                // perspective divide in the 4th dimension
      proj[k].set(p[0] * f, p[1] * f, p[2] * f);
      Vp[k*3] = proj[k].x; Vp[k*3+1] = proj[k].y; Vp[k*3+2] = proj[k].z;
    }
    edges.forEach(([a, b], e) => {
      P[e*6] = proj[a].x; P[e*6+1] = proj[a].y; P[e*6+2] = proj[a].z;
      P[e*6+3] = proj[b].x; P[e*6+4] = proj[b].y; P[e*6+5] = proj[b].z;
    });
    eGeo.attributes.position.needsUpdate = vGeo.attributes.position.needsUpdate = true;
  };
  born = performance.now() - REVEAL - 1; G.userData.tick();   // seed at full size so fitToView frames the finished tesseract
  eGeo.computeBoundingBox(); eGeo.computeBoundingSphere();
  born = performance.now();                                   // ...then start the reveal for real
  return G;
}
// The three regular 4-polytopes cheap enough to generate by rule — every edge falls straight out of the
// vertex coordinates, so there's no hand-typed edge list to get wrong.
function makeTesseract() {                       // 8-cell: 16 verts at (±1,±1,±1,±1); edge = differ in exactly one axis
  const v = []; for (let i = 0; i < 16; i++) v.push([i & 1 ? 1 : -1, i & 2 ? 1 : -1, i & 4 ? 1 : -1, i & 8 ? 1 : -1]);
  const e = []; for (let i = 0; i < 16; i++) for (let j = i + 1; j < 16; j++) {
    let d = 0; for (let k = 0; k < 4; k++) if (v[i][k] !== v[j][k]) d++; if (d === 1) e.push([i, j]);
  }
  return make4D(v, e);                           // 32 edges
}
function makeCell16() {                          // 16-cell: 8 verts at ±eₖ; edge between every pair but the antipodes
  const v = []; for (let k = 0; k < 4; k++) { const a = [0, 0, 0, 0], b = [0, 0, 0, 0]; a[k] = 1; b[k] = -1; v.push(a, b); }
  const e = []; for (let i = 0; i < 8; i++) for (let j = i + 1; j < 8; j++) {
    let anti = true; for (let k = 0; k < 4; k++) if (v[i][k] !== -v[j][k]) anti = false; if (!anti) e.push([i, j]);
  }
  return make4D(v, e);                           // 24 edges
}
function makeCell5() {                            // 5-cell (4-simplex): tetrahedron base + apex on the w-axis, all 10 pairs joined
  const r = 1 / Math.sqrt(5), v = [[1, 1, 1, -r], [1, -1, -1, -r], [-1, 1, -1, -r], [-1, -1, 1, -r], [0, 0, 0, 4 * r]];
  const e = []; for (let i = 0; i < 5; i++) for (let j = i + 1; j < 5; j++) e.push([i, j]);
  return make4D(v, e);                            // 10 edges
}
const MODELS = { rocket: makeRocket, engine: makeEngine, satellite: makeSatellite, molecule: makeMolecule, drone: makeDrone,
  tesseract: makeTesseract, cell16: makeCell16, cell5: makeCell5 };
let currentModelKey = 'rocket';   // which demo is loaded (null once the user loads their own file) — Share view encodes it
let model = makeRocket();
pivot.add(model);

function fitToView(obj) { // drop any model centered on the stage with its feet on the floor
  obj.position.set(0, 0, 0); obj.scale.setScalar(1); obj.rotation.set(0, 0, 0);
  obj.updateMatrixWorld(true);                       // measure the raw geometry extent, transform reset
  const box = new THREE.Box3().setFromObject(obj);
  obj.userData.dims = box.getSize(new THREE.Vector3());   // authored extent (transform reset => real units, mm for CAD) — Measure mode reads this
  obj.userData.dcen = box.getCenter(new THREE.Vector3());
  const t = fitTransform(box.min, box.max, 1.8, FLOOR_Y); // feet on the baseline floor; Ground slider rides it up/down
  obj.scale.setScalar(t.scale);
  obj.position.set(t.position.x, t.position.y, t.position.z);
}
fitToView(model);

// Parts = the pieces the explode and point-to-inspect gestures act on. Downloaded models never hand
// you those as the top-level children: Sketchfab wraps everything in a chain of single-child nodes
// (Sketchfab_model ▸ LEGO.fbx ▸ Object_2 ▸ RootNode ▸ …) and often bolts on extras like a floor
// plane, so the direct children are one lone group, "Parts" reads 1, and nothing explodes. Scan the
// whole tree and take the node with the MOST mesh-bearing children — that's the assembly level.
// Shallowest wins ties (traverse is pre-order), which keeps hand-built models like the rocket intact.
// ponytail: O(n²) since firstMesh re-walks per child; runs once per load, memoize if a model drags.
let parts = [], partSet = new Set();
let partLabelEls = [];   // pooled name-chips, one per part; declared here so the first collectParts (init) can fill it
const removed = [];    // parts pulled out and dissolved by the removal gesture — Restore brings them back
const ejecting = [];   // { p, t, pull0 } mid fly-out-and-fade animation
function firstMesh(o) { let m = null; o.traverse(c => { if (!m && c.isMesh) m = c; }); return m; }
function partRoot(root) {
  let best = root, bestN = -1;
  root.traverse(o => { const n = o.children.filter(firstMesh).length; if (n > bestN) { best = o; bestN = n; } });
  return best;
}
function collectParts(root) {
  const host = partRoot(root);
  parts = host.children.filter(firstMesh);
  if (parts.length < 2) { parts = []; root.traverse(c => { if (c.isMesh) parts.push(c); }); } // last resort: every mesh
  partSet = new Set(parts);
  // Explode direction per part: model center -> part center, measured from bounding boxes rather
  // than node origins, because exporters routinely bake transforms so every node sits at (0,0,0)
  // and those parts would have nowhere to travel. Taken in each part's OWN parent space, which is
  // the space .position lives in. A part sitting dead-center gets a zero direction and stays put.
  root.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(host);
  const mid = box.getCenter(new THREE.Vector3());
  const span = box.getSize(new THREE.Vector3()).length() || 1;
  parts.forEach((p, i) => {
    p.userData.pid = i;                       // stable handle for the removal state machine
    p.userData.pull = p.userData.pullT = 0;   // how far it's been pulled out: eased value, and its target
    p.userData.carry = new THREE.Vector3();   // grab offset, eased as a WHOLE vector so a turn toward the bin can't teleport the part
    p.userData.home = p.position.clone();
    const c = new THREE.Box3().setFromObject(p).getCenter(new THREE.Vector3());
    p.userData.coff = p.worldToLocal(c.clone());   // part's geometric centre in its OWN local space — constant as it moves rigidly, for screen-space proximity
    p.userData.dir = p.parent.worldToLocal(c).sub(p.parent.worldToLocal(mid.clone()));
    // Pulling a part out needs somewhere to pull it, and a dead-center part (a gearbox housing,
    // say) has a zero explode direction by design. Give those one toward the viewer so they can
    // still be extracted, without making them drift during a normal explode.
    p.userData.pdir = p.userData.dir.lengthSq() > 1e-8 ? p.userData.dir : new THREE.Vector3(0, 0, span * 0.35);
    if (!p.userData.core) p.userData.core = firstMesh(p);
    // Number + name. The number is the speakable handle ("delete part 3") and is ALWAYS shown; the name
    // is the authored one if the model carries a meaningful one (rocket's "Fuel Tank", a well-named GLB
    // mesh), captured ONCE so a re-collect can't stack "3. 3. …". ponytail: no fake specs beyond the name.
    if (p.userData.name === undefined) p.userData.name = p.userData.label ?? (p.name && p.name !== 'Part' ? p.name : null);
    p.userData.num = i + 1;
    p.userData.label = p.userData.name ? `${p.userData.num}. ${p.userData.name}` : `Part ${p.userData.num}`;
    p.userData.info = `component ${p.userData.num} of ${parts.length}`;
  });
  applyClip(root);   // every material has to carry the section plane, or half the model ignores the cut
  buildPartLabels(); // one floating name-chip per part, shown while exploded
}
collectParts(model);

// ---------- render modes: Normal · Holo · X-ray — re-dress ANY model in one of three looks ----------
// Uploaded models arrive with their own materials. Three modes the user switches between:
//   normal — the model's real materials/textures, exactly as authored (a GLB's true colours).
//   holo   — translucent emissive-blue core + cyan wireframe overlay: the JARVIS look (the default).
//   xray   — additive, double-sided, see-through emissive: overlapping surfaces stack into a bright
//            "scan" glow and edges read hot. Reads great on the dark stage; strong for CAD assemblies.
// The authored material is cached the first time so Normal can restore it. The built-in rocket
// (holoNative) follows the mode too, so all three demo before you upload anything — its own blue core
// is its normal/holo fill and only the wireframe overlay differs. holoWire overlays (rocket + skinned
// uploads) show only in Holo. ponytail: a 4th mode is just one more makeXxxMat + a button + a list entry.
const RENDER_MODES = ['normal', 'holo', 'xray'];
// Bloom is a hologram effect, not a lighting fix: a realistic Normal render (like a real product photo)
// shouldn't halo at all, or a shiny uploaded model glows white. Holo/X-ray are meant to glow, so keep it
// there. applyRenderMode drives bloom.strength off this; the Bloom checkbox toggles bloom.enabled on top.
const MODE_BLOOM = { normal: 0.12, holo: 0.7, xray: 0.55 };
// Per-mode backdrop. Holo keeps the flat near-black void (it reads best behind glowing blue wire — the
// user's call). Normal gets a soft studio "spotlight pool" so a realistic/textured model sits in a product
// shot instead of on a dead black card; X-ray gets a cool teal scanner glow to match the translucent cyan.
// Painted as radial-gradient canvas textures (no image assets, no build step). They're tone-mapped like the
// rest of the frame, so the inner colours are picked a touch hot to survive ACES at 0.9 exposure.
function gradientBG(inner, outer) {
  const cv = document.createElement('canvas'); cv.width = cv.height = 512;
  const g = cv.getContext('2d');
  const rg = g.createRadialGradient(256, 300, 24, 256, 300, 430);   // bright core low-centre (behind the model), fading out to the frame
  rg.addColorStop(0, inner); rg.addColorStop(1, outer);
  g.fillStyle = rg; g.fillRect(0, 0, 512, 512);
  const tex = new THREE.CanvasTexture(cv); tex.colorSpace = THREE.SRGBColorSpace; return tex;
}
const MODE_BG = {
  normal: gradientBG('#46566b', '#06080d'),   // studio: cool steel pool -> near-black vignette
  holo:   new THREE.Color(0x0a0a0f),           // unchanged void
  xray:   gradientBG('#12586a', '#02070c'),    // scanner: teal-cyan glow -> black
};
let renderMode = 'holo', autoSpin = true;   // 'holo' == the previous default, so the boot look is unchanged
let lockX = false, lockY = false, lockZ = false; // freeze a rotation axis so the model spins cleanly around the free one(s)
let offsetX = 0; // slide the model sideways off the projector center (scene units), set by the Move X slider
const makeHoloMat = () => new THREE.MeshStandardMaterial({
  color: 0x0a2a4a, emissive: 0x0aa0ff, emissiveIntensity: BASE_EMISSIVE,
  metalness: 0.3, roughness: 0.35, transparent: true, opacity: 0.82 });
const makeXrayMat = () => new THREE.MeshStandardMaterial({          // additive + double-side => density glow
  // Additive means every overlapping layer ADDS light, so a dense model (a whole Iron Man suit is dozens of
  // stacked plates) piles up to solid white. Keep the per-layer contribution small (intensity*opacity) so
  // thin areas stay see-through and only genuinely dense stacks glow hot — tone mapping then rolls the
  // densest cores to a bright cyan-white rim instead of a flat white blob. ponytail: tune these two if a
  // typical model reads too faint (raise) or still whites out (lower).
  color: 0x001824, emissive: 0x3fd8ff, emissiveIntensity: 0.4,
  metalness: 0, roughness: 1, transparent: true, opacity: 0.15,
  depthWrite: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending });
function applyRenderMode(root, mode) {
  root.traverse(o => {
    if (!o.isMesh || o.userData.holoWire) return;              // overlay wires are toggled in the second pass
    if (!o.userData.origMat) o.userData.origMat = o.material;  // cache the authored material once, for Normal
    const native = o.userData.holoNative;                      // the rocket's own blue core is its normal/holo fill
    if (mode === 'holo' && !native) {
      if (!o.userData.holoMat) o.userData.holoMat = makeHoloMat();
      if (!o.userData.wire) {                                  // cyan wireframe overlay, built on first Holo
        const w = new THREE.Mesh(o.geometry, new THREE.MeshBasicMaterial({
          color: 0x66e0ff, wireframe: true, transparent: true, opacity: 0.28 }));
        w.userData.holoWire = true; w.raycast = () => {};      // never intercept point-to-inspect rays
        o.add(w); o.userData.wire = w;
      }
    }
    if (mode === 'xray' && !o.userData.xrayMat) o.userData.xrayMat = makeXrayMat();
    o.material = mode === 'xray'              ? o.userData.xrayMat
               : (mode === 'holo' && !native) ? o.userData.holoMat
               :                                o.userData.origMat;
  });
  root.traverse(o => { if (o.userData.holoWire) o.visible = (mode === 'holo'); });   // rocket + upload wires: Holo only
  bloom.strength = MODE_BLOOM[mode];   // Normal barely blooms (realistic); Holo/X-ray glow
  scene.background = MODE_BG[mode];    // studio pool / void / scanner glow per mode
  applyClip(root);   // fresh materials arrive without the section plane
}

// ---------- hand skeleton drawn INTO the 3D scene (your hand appears inside the hologram) ----------
const MAX_JOINTS = 42;               // 2 hands * 21 landmarks
const MAX_BONES = 2 * 24;            // 2 hands * (HAND_CONNECTIONS is 21, 24 is safe headroom)
const joints = new THREE.InstancedMesh(
  new THREE.SphereGeometry(0.045, 12, 12),
  new THREE.MeshBasicMaterial({ color: 0xffffff }), MAX_JOINTS);   // per-instance color set in updateHandViz
joints.frustumCulled = false; scene.add(joints);
const JOINT_LIVE = new THREE.Color(0x7CFFB2), JOINT_PARK = new THREE.Color(0xff5a5a); // green = open/live, red = a closed fist (gripping)
const boneGeo = new THREE.BufferGeometry();
boneGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(MAX_BONES * 2 * 3), 3));
const bones = new THREE.LineSegments(boneGeo, new THREE.LineBasicMaterial({ color: 0x7CFFB2, transparent: true, opacity: 0.85 }));
bones.frustumCulled = false; scene.add(bones);

// Fingertip comet-trails: a short additive line trailing each hand's index tip, fading tail->tip (JARVIS feel).
const TRAIL_LEN = 18, TRAIL_RGB = [0.4, 0.9, 1.0];
const trails = [0, 1].map(() => {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(TRAIL_LEN * 3), 3));
  g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(TRAIL_LEN * 3), 3));
  const line = new THREE.Line(g, new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false }));
  line.frustumCulled = false; scene.add(line);
  return { line, hist: [] };
});
function updateTrail(slot, tip) {            // tip = world {x,y,z}, or null to shrink the tail away
  const t = trails[slot]; if (!t) return;
  if (tip) { t.hist.push(tip); while (t.hist.length > TRAIL_LEN) t.hist.shift(); }
  else if (t.hist.length) t.hist.shift();
  const n = t.hist.length, P = t.line.geometry.attributes.position.array, C = t.line.geometry.attributes.color.array;
  for (let k = 0; k < n; k++) {
    const p = t.hist[k], f = n > 1 ? k / (n - 1) : 1;    // 0 at tail, 1 at fingertip
    P[k*3] = p.x; P[k*3+1] = p.y; P[k*3+2] = p.z;
    C[k*3] = TRAIL_RGB[0]*f; C[k*3+1] = TRAIL_RGB[1]*f; C[k*3+2] = TRAIL_RGB[2]*f;
  }
  t.line.geometry.setDrawRange(0, n);
  t.line.geometry.attributes.position.needsUpdate = true;
  t.line.geometry.attributes.color.needsUpdate = true;
}
const _m = new THREE.Matrix4(), _hidden = new THREE.Matrix4().makeScale(0, 0, 0);
function hideHands() {
  for (let k = 0; k < MAX_JOINTS; k++) joints.setMatrixAt(k, _hidden);
  joints.instanceMatrix.needsUpdate = true; boneGeo.setDrawRange(0, 0);
  for (const t of trails) { t.hist.length = 0; t.line.geometry.setDrawRange(0, 0); }
}
function updateHandViz(hands) {
  const conns = HandLandmarker.HAND_CONNECTIONS, pos = boneGeo.attributes.position.array;
  let j = 0, v = 0, hi = 0; // joint index, bone-vertex index, hand slot
  for (const hand of hands) {
    const world = hand.map(p => landmarkToWorld(p, HAND_SPAN, HAND_DEPTH));
    const col = handPose(hand) === 'open' ? JOINT_PARK : JOINT_LIVE;     // an open hand is ignored (neutral) → dim red; any active pose glows live green
    for (const p of world) if (j < MAX_JOINTS) { _m.makeTranslation(p.x, p.y, p.z); joints.setMatrixAt(j, _m); joints.setColorAt(j, col); j++; }
    for (const c of conns) if (v + 2 <= MAX_BONES * 2) {
      const a = world[c.start], d = world[c.end];
      pos.set([a.x, a.y, a.z, d.x, d.y, d.z], v * 3); v += 2;
    }
    updateTrail(hi++, world[INDEX_TIP]);
  }
  for (let s = hi; s < trails.length; s++) updateTrail(s, null); // fade out trails for absent hands
  for (let k = j; k < MAX_JOINTS; k++) joints.setMatrixAt(k, _hidden); // hide unused instances
  joints.instanceMatrix.needsUpdate = true;
  if (joints.instanceColor) joints.instanceColor.needsUpdate = true;
  boneGeo.setDrawRange(0, v); boneGeo.attributes.position.needsUpdate = true;
}
hideHands();

function resize() {
  if (!innerWidth || !innerHeight) return;   // a collapsed/0-sized pane would give aspect = NaN and a zero-size framebuffer
  renderer.setSize(innerWidth, innerHeight, false);
  composer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
}
addEventListener('resize', resize); resize();

// ---------- load your own model (object URL avoids any CORS/backend) ----------
// Pick a loader by file extension and normalize its result to an Object3D we can drop in.
// STL carries only geometry, so we wrap it in a holo-styled mesh. .blend is Blender's own
// binary format — no browser can read it; the standard path is to export glTF from Blender.
const LOADERS = {
  glb:  { L: GLTFLoader, pick: r => r.scene },
  gltf: { L: GLTFLoader, pick: r => r.scene },
  obj:  { L: OBJLoader,  pick: r => r },
  fbx:  { L: FBXLoader,  pick: r => r },
  stl:  { L: STLLoader,  pick: g => new THREE.Mesh(g, new THREE.MeshStandardMaterial({
           color: 0x143a5a, emissive: 0x0aa0ff, emissiveIntensity: BASE_EMISSIVE, metalness: 0.3, roughness: 0.35 })) },
};

// ---------- CAD import: STEP / IGES via occt-import-js (OpenCASCADE, compiled to WASM) ----------
// The loaders above read files that are ALREADY triangles (STL/OBJ/glTF). A real CAD file (.step/.stp/
// .iges) is a B-rep — parametric surfaces, no triangles — so no three.js loader can touch it; it has to be
// tessellated first. occt-import-js is OpenCASCADE's kernel compiled to WASM, and it returns one named mesh
// PER SOLID, so a STEP assembly becomes one grabbable/inspectable part per component (an STL, by contrast,
// is a single blob = one part). The 7.6 MB WASM is fetched from the CDN only on the FIRST CAD import, so
// glTF/STL users never pay for it. ponytail: pinned to @0.0.23; bump when occt-import-js releases.
const OCCT_JS   = 'https://cdn.jsdelivr.net/npm/occt-import-js@0.0.23/dist/occt-import-js.js';
const OCCT_WASM = 'https://cdn.jsdelivr.net/npm/occt-import-js@0.0.23/dist/occt-import-js.wasm';
const CAD_READ  = { step: 'ReadStepFile', stp: 'ReadStepFile', iges: 'ReadIgesFile', igs: 'ReadIgesFile' };
let _occt = null;
function occtReady() {                       // load the script tag + init the WASM once, lazily
  if (!_occt) _occt = new Promise((res, rej) => {
    const s = document.createElement('script');
    s.src = OCCT_JS; s.onload = res; s.onerror = () => rej(new Error('could not fetch the CAD engine'));
    document.head.appendChild(s);
  }).then(() => window.occtimportjs({ locateFile: () => OCCT_WASM }));   // point Emscripten at the CDN .wasm
  return _occt;
}
// Rebuild occt's flat position/normal/index arrays as one three.js mesh per solid, grouped so collectParts
// sees one part per solid. Names + colours ride along (the label, and the real colour when holo-skin is off).
function occtToGroup(result) {
  if (!result || !result.success) throw new Error('the CAD file could not be read');
  const group = new THREE.Group();
  result.meshes.forEach((m, i) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(m.attributes.position.array, 3));
    if (m.attributes.normal) g.setAttribute('normal', new THREE.Float32BufferAttribute(m.attributes.normal.array, 3));
    g.setIndex(new THREE.BufferAttribute(new Uint32Array(m.index.array), 1));   // Uint32: CAD solids blow past 65 k verts
    if (!m.attributes.normal) g.computeVertexNormals();
    const col = m.color ? new THREE.Color(m.color[0], m.color[1], m.color[2]) : new THREE.Color(0x143a5a);
    const mesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: col, metalness: 0.35, roughness: 0.45 }));
    mesh.name = m.name || `Solid ${i + 1}`;
    group.add(mesh);
  });
  return group;
}
async function loadCAD(file, reader) {
  const occt = await occtReady();
  return occtToGroup(occt[reader](new Uint8Array(await file.arrayBuffer()), null));
}
// Hand the old model's GPU memory back before dropping it. Nothing here is small — the models
// people actually load run 8-120 MB — so swapping three or four without this quietly fills VRAM
// and the frame rate falls off. Materials are collected rather than traversed because holo-skin
// parks the original alongside its replacement in userData; textures hang off the materials.
function disposeTree(root) {
  root.traverse(o => {
    if (!o.isMesh) return;
    o.geometry.dispose();
    for (const m of [o.material, o.userData.origMat, o.userData.holoMat].flat()) {
      if (!m) continue;
      for (const v of Object.values(m)) if (v?.isTexture) v.dispose();
      m.dispose();
    }
  });
}
function swapModel(obj) {
  clearHighlight();                 // before the dispose: the highlight holds a material from the OLD model
  // Drop the removal state rather than restoring it — these parts belong to the model being thrown
  // away, and `pid` indices into the new `parts` array would point at whatever now sits there.
  removed.length = 0; ejecting.length = 0; aim = AIM_OFF; locked = null; grabTip = null; binShow = false;
  currentModelKey = null;   // a swap clears "which demo"; loadDemo re-sets it right after, a file load leaves it null
  for (const b of document.querySelectorAll('.demo')) b.classList.remove('on');
  pivot.remove(model); disposeTree(model);
  model = obj; fitToView(model); pivot.add(model);
  $('boot')?.classList.add('hidden');   // loading any model (demo or file) reveals the stage — not just Start camera
  collectParts(model); applyRenderMode(model, renderMode);   // dress the new model in the current mode
  if (measureOn) buildMeasure();   // rebuild the dimension cage for the new model's authored box
  // Reset scale too, not just rotation: fitToView just sized this model to the stage, and leaving a
  // previous 3.5x zoom on the pivot multiplies straight back through it and throws it off-screen.
  target.explode = current.explode = 0; target.rx = target.ry = target.rz = 0; target.scale = 1; spinVel.rx = spinVel.ry = 0;
  updateStats(0);   // refresh the parts count for the newly loaded model (live tracking re-fills hands next frame)
}
$('file').addEventListener('change', (e) => {
  const f = e.target.files[0]; if (!f) return;
  initAudio();
  const ext = (f.name.split('.').pop() || '').toLowerCase();
  if (ext === 'blend') {
    errEl.textContent = ".blend can't be read in the browser. In Blender: File ▸ Export ▸ glTF 2.0 (.glb), then load that file.";
    e.target.value = ''; return;
  }
  if (CAD_READ[ext]) {                                                       // .step/.stp/.iges: tessellate via WASM (async) then swap in
    errEl.textContent = ''; stateEl.textContent = `Tessellating ${f.name}… (first CAD load fetches the engine, ~7 MB)`;
    loadCAD(f, CAD_READ[ext])
      .then((group) => { swapModel(group); stateEl.textContent = `Loaded ${f.name}`; blip(720, 0.14, 'triangle'); })
      .catch((err) => { errEl.textContent = 'Could not load CAD model: ' + (err.message || err); })
      .finally(() => { e.target.value = ''; });
    return;
  }
  const entry = LOADERS[ext];
  if (!entry) { errEl.textContent = `Unsupported file “.${ext}” — use .glb, .gltf, .obj, .fbx, .stl, .step or .iges`; e.target.value = ''; return; }
  errEl.textContent = ''; stateEl.textContent = `Loading ${f.name}…`;
  const url = URL.createObjectURL(f);
  new entry.L().load(url, (res) => {
    swapModel(entry.pick(res)); URL.revokeObjectURL(url);
    stateEl.textContent = `Loaded ${f.name}`; blip(720, 0.14, 'triangle');
    e.target.value = '';   // clear it, or re-picking the SAME file fires no change event and looks broken
  }, undefined, (err) => { URL.revokeObjectURL(url); e.target.value = ''; errEl.textContent = 'Could not load model: ' + err; });
});

// Built-in demo gallery: procedural models build instantly (offline); heavier GLB demos stream in
// lazily on click. Both route through the same swapModel path a file load uses, so explode / inspect /
// labels all work identically no matter where the model came from.
const FILE_DEMOS = {   // nothing downloads until you actually pick one of these
  reactor: 'models/arc_reactor.glb', ironman: 'models/iron-man_mark_7.glb',
  taj: 'models/taj_mahal.glb', gears: 'models/gears_animation.glb',
};
const _lightDemo = (key) => { for (const b of document.querySelectorAll('.demo')) b.classList.toggle('on', b.dataset.model === key); };
function loadDemo(key, after) {
  initAudio(); blip(680, 0.09, 'triangle');
  if (MODELS[key]) {                        // procedural: build + swap synchronously, offline, instant
    swapModel(MODELS[key]()); currentModelKey = key; _lightDemo(key);   // swapModel nulls the key; we own it again
    stateEl.textContent = `Loaded demo · ${key}`; after?.(); return;
  }
  const url = FILE_DEMOS[key]; if (!url) return;
  _lightDemo(key); errEl.textContent = '';  // light the button now; the file streams in behind a loading note
  stateEl.textContent = `Loading ${key}…`;
  new GLTFLoader().load(url, (res) => {
    swapModel(res.scene); currentModelKey = key; _lightDemo(key);
    if (res.animations.length) {            // e.g. the gears rig — drive its clips through the per-frame model.userData.tick hook
      const clock = new THREE.Clock(), mixer = new THREE.AnimationMixer(res.scene);
      res.animations.forEach((c) => mixer.clipAction(c).play());
      res.scene.userData.tick = () => mixer.update(clock.getDelta());
    }
    stateEl.textContent = `Loaded demo · ${key}`; blip(720, 0.14, 'triangle'); after?.();   // re-apply a shared view AFTER swapModel's transform reset
  }, undefined, (err) => { errEl.textContent = `Could not load ${key}: ` + err; stateEl.textContent = ''; });
}
for (const b of document.querySelectorAll('.demo')) b.addEventListener('click', () => loadDemo(b.dataset.model));
document.querySelector('.demo[data-model="rocket"]')?.classList.add('on');   // rocket is the boot model

// ---------- HUD controls: render mode · showcase toggles · snapshot · live calibration ----------
// Segmented Normal/Holo/X-ray control. setRenderMode lights the active button, drops any highlight
// (its cached emissive belongs to the old material), and re-dresses the model. Also driven by the M key.
const MODE_BTN = { normal: $('mNormal'), holo: $('mHolo'), xray: $('mXray') };
function setRenderMode(mode) {
  if (!RENDER_MODES.includes(mode)) return;
  renderMode = mode;
  for (const k of RENDER_MODES) MODE_BTN[k].classList.toggle('on', k === mode);
  clearHighlight();                 // the old material's cached emissive doesn't carry to the new one
  applyRenderMode(model, mode);
}
for (const k of RENDER_MODES)
  MODE_BTN[k].addEventListener('click', () => { setRenderMode(k); blip(k === 'xray' ? 820 : 560, 0.05); });
$('tSpin').addEventListener('change', (e) => { autoSpin = e.target.checked; });
$('tBloom').addEventListener('change', (e) => { bloom.enabled = e.target.checked; });
$('tGrid').addEventListener('change', (e) => { grid.visible = reticle.visible = e.target.checked; });
$('snap').addEventListener('click', () => {                 // one-tap PNG of the holo render (preserveDrawingBuffer)
  initAudio(); blip(900, 0.08, 'triangle');
  const a = document.createElement('a');
  a.download = 'origin-' + Date.now() + '.png';
  a.href = renderer.domElement.toDataURL('image/png'); a.click();
});
$('secBtn').addEventListener('click', () => setSection(!sectionOn));
$('tuneBtn').addEventListener('click', () => { initAudio(); $('tune').classList.toggle('hidden'); });

// Camera PIP: drawHands() already paints the skeleton onto #overlay every frame — we just reveal the box.
$('tCam').addEventListener('change', (e) => {
  $('cam').classList.toggle('show', e.target.checked);
  if (e.target.checked && !running) stateEl.textContent = 'Camera view on — click ▶ Start camera to see the feed';
});
// Auto-play: hands-free attract loop — eases explode 0→1→0 forever (driven in loop()). Great for a booth / video.
let playing = false, playT = 0;
$('tPlay').addEventListener('change', (e) => { playing = e.target.checked; if (playing) { autoSpin = true; $('tSpin').checked = true; } });
// Present: fade the HUD away and go fullscreen so the hologram fills the screen (also the F key).
let presenting = false;
function setPresent(on) {
  presenting = on; document.body.classList.toggle('present', on); $('presentBtn').classList.toggle('on', on);
  if (on) document.documentElement.requestFullscreen?.().catch(() => {});
  else if (document.fullscreenElement) document.exitFullscreen?.().catch(() => {});
  initAudio(); blip(on ? 720 : 360, 0.08);
}
$('presentBtn').addEventListener('click', () => setPresent(!presenting));
document.addEventListener('fullscreenchange', () => { if (!document.fullscreenElement && presenting) setPresent(false); });   // Esc out of fullscreen leaves present too
// Share view: copy a link that reproduces the current demo + rotation + zoom + explode + render mode.
function stateToHash() {
  const p = new URLSearchParams();
  if (currentModelKey) p.set('m', currentModelKey);
  p.set('rx', current.rx.toFixed(3)); p.set('ry', current.ry.toFixed(3)); p.set('rz', current.rz.toFixed(3));
  p.set('s', current.scale.toFixed(3)); p.set('e', current.explode.toFixed(3)); p.set('mode', renderMode);
  return '#' + p.toString();
}
$('shareBtn').addEventListener('click', async () => {
  const url = location.origin + location.pathname + stateToHash();
  const btn = $('shareBtn'), label = btn.dataset.label || (btn.dataset.label = btn.textContent);
  const flash = (msg) => {                                   // flash the BUTTON itself — the status line sits too far away to read as feedback
    btn.textContent = msg; btn.classList.add('ok');
    stateEl.textContent = `🔗 ${msg} — paste the link to reopen this exact view`;
    clearTimeout(btn._t); btn._t = setTimeout(() => { btn.textContent = label; btn.classList.remove('ok'); }, 1600);
  };
  try { await navigator.clipboard.writeText(url); flash('✓ Link copied'); }
  catch { location.hash = stateToHash(); flash('✓ Link in address bar'); }   // clipboard blocked (insecure origin / denied): the hash IS the shareable URL
  initAudio(); blip(880, 0.08, 'triangle');
});
function bindRange(id, apply, outId, fmt) {                 // wire a slider to a live tuning variable
  const el = $(id), out = $(outId);
  const upd = () => { const v = parseFloat(el.value); apply(v); out.textContent = fmt(v); };
  el.addEventListener('input', upd); upd();
}
bindRange('sRot', (v) => ROT_SPEED = v, 'vRot', (v) => v.toFixed(1));
bindRange('sZoom', (v) => ZOOM_SPEED = v, 'vZoom', (v) => v.toFixed(1));
bindRange('sSmooth', (v) => SMOOTH = v, 'vSmooth', (v) => v.toFixed(2));
bindRange('sGround', (v) => setGround(v), 'vGround', (v) => v.toFixed(2));       // raise/lower the base the model stands on
bindRange('sOffX', (v) => { offsetX = v; pivot.position.x = v; }, 'vOffX', (v) => v.toFixed(1)); // slide the model along X
$('lockX').addEventListener('change', (e) => lockX = e.target.checked);          // freeze pitch, roll, or yaw for a clean turntable
$('lockY').addEventListener('change', (e) => lockY = e.target.checked);
$('lockZ').addEventListener('change', (e) => lockZ = e.target.checked);
$('tMirror').addEventListener('change', (e) => { MIRROR_X = e.target.checked ? -1 : 1; });

// ---------- transform state: gestures set `target`, each frame eases `current` toward it ----------
const target = { rx: 0, ry: 0, rz: 0, scale: 1, explode: 0 };
const current = { rx: 0, ry: 0, rz: 0, scale: 1, explode: 0 };
$('reset').addEventListener('click', () => { initAudio(); blip(420, 0.12, 'sine'); target.rx = 0; target.ry = 0; target.rz = 0; target.scale = 1; target.explode = 0; spinVel.rx = spinVel.ry = 0; restoreParts(); $('sOffX').value = 0; $('sOffX').dispatchEvent(new Event('input')); }); // recenter on X and put removed parts back (Ground stays: it's stage calibration)

// ---------- mouse & keyboard fallback ----------
// The demo can't depend on a webcam. Someone with no camera — or who dismisses the permission
// prompt — would otherwise get an auto-spinning rocket and no way in at all. These write the SAME
// `target` the gestures write, so there's no second code path downstream, and they stay live during
// tracking, which is how you hold a model still for a screenshot.
let manualT = 0;                       // hold off the idle turntable for a moment after any manual input
const manual = () => { manualT = performance.now() + 2500; };
const cv = renderer.domElement;
let drag0 = null;
cv.addEventListener('pointerdown', (e) => { drag0 = { x: e.clientX, y: e.clientY }; cv.setPointerCapture(e.pointerId); manual(); });
cv.addEventListener('pointerup', () => { drag0 = null; dragging = false; });
cv.addEventListener('pointermove', (e) => {
  if (!drag0) return;
  // Both axes divide by innerHeight so a diagonal drag turns the model diagonally instead of
  // skewing with the window's aspect ratio.
  const dry = (e.clientX - drag0.x) / innerHeight * 3, drx = (e.clientY - drag0.y) / innerHeight * 3;
  target.ry += dry; target.rx += drx;
  spinVel.ry = spinVel.ry * (1 - SPIN_CAPTURE) + dry * SPIN_CAPTURE;   // same flick-to-spin as a pinch drag
  spinVel.rx = spinVel.rx * (1 - SPIN_CAPTURE) + drx * SPIN_CAPTURE;
  drag0 = { x: e.clientX, y: e.clientY }; dragging = true; manual();
});
cv.addEventListener('wheel', (e) => {
  e.preventDefault();
  target.scale = clamp(target.scale * (1 - e.deltaY * 0.0015), SCALE_MIN, SCALE_MAX); manual();
}, { passive: false });
const zoomBy = (f) => target.scale = clamp(target.scale * f, SCALE_MIN, SCALE_MAX);
const KEYS = {                          // arg is the rotation step (bigger with Shift)
  ArrowLeft: (s) => target.ry -= s, ArrowRight: (s) => target.ry += s,
  ArrowUp:   (s) => target.rx -= s, ArrowDown:  (s) => target.rx += s,
  '[': () => target.explode = clamp(target.explode - 0.1, 0, 1),
  ']': () => target.explode = clamp(target.explode + 0.1, 0, 1),
  '=': () => zoomBy(1.1), '+': () => zoomBy(1.1), '-': () => zoomBy(1 / 1.1),
  c: () => setSection(!sectionOn),
  r: () => $('reset').click(),
  x: () => $('restore').click(),
  p: () => $('snap').click(),
  m: () => setRenderMode(RENDER_MODES[(RENDER_MODES.indexOf(renderMode) + 1) % RENDER_MODES.length]),  // cycle Normal→Holo→X-ray
  d: () => { $('tDims').checked = !$('tDims').checked; setMeasure($('tDims').checked); },   // toggle dimensions
  v: () => $('voiceBtn').click(),                                                            // toggle voice control
  f: () => $('presentBtn').click(),                                                          // present / fullscreen (a keydown is a valid fullscreen gesture)
};
addEventListener('keydown', (e) => {
  // Let the browser have its own chords, and don't steal keys from a focused control (the sliders
  // are arrow-key operated, and Space/Enter on a focused button must still press it).
  if (e.metaKey || e.ctrlKey || e.altKey || e.target?.closest?.('input, select, textarea, button')) return;
  const fn = KEYS[e.key.length === 1 ? e.key.toLowerCase() : e.key];
  if (!fn) return;
  e.preventDefault(); fn(e.shiftKey ? 0.3 : 0.1); manual();
});

// ---------- voice commands: hands-free control by speech (Web Speech API) ----------
// Another LOCAL browser sensor, exactly like the camera: the recognition runs in the browser, and all we
// do is map a final transcript to the SAME action a button/key already calls. No LLM, no key, no request
// from us. Talk OR gesture — say "x-ray", "explode", "measure" while your hands drive the model.
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
const VOICE = [   // [words that trigger it] -> action; first hit wins, so list the specific before the generic
  [['normal'],                        () => setRenderMode('normal')],
  [['x-ray', 'x ray', 'xray', 'exray'], () => setRenderMode('xray')],
  [['holo', 'hologram'],              () => setRenderMode('holo')],
  [['explode', 'apart', 'blow'],      () => { target.explode = 1; }],
  [['assemble', 'rebuild', 'together', 'collapse'], () => { target.explode = 0; }],
  [['measure', 'dimension', 'size'],  () => { $('tDims').checked = !$('tDims').checked; setMeasure($('tDims').checked); }],
  [['section', 'cut', 'slice'],       () => setSection(!sectionOn)],
  [['bigger', 'zoom in', 'closer'],   () => zoomBy(1.25)],
  [['smaller', 'zoom out', 'further'],() => zoomBy(1 / 1.25)],
  [['spin', 'rotate'],                () => { autoSpin = true; $('tSpin').checked = true; }],
  [['stop', 'freeze', 'still'],       () => { autoSpin = false; $('tSpin').checked = false; }],
  [['snapshot', 'capture', 'photo'],  () => $('snap').click()],
  [['restore', 'undo'],               () => $('restore').click()],
  [['reset', 'center', 'home'],       () => $('reset').click()],
  // load a model by name — hands-free demo switching, the payoff for a live/video walkthrough
  [['rocket'],                        () => loadDemo('rocket')],
  [['turbofan', 'engine', 'jet'],     () => loadDemo('engine')],
  [['reactor', 'arc reactor'],        () => loadDemo('reactor')],
  [['iron man', 'ironman'],           () => loadDemo('ironman')],
  [['taj', 'mahal'],                  () => loadDemo('taj')],
  [['gears', 'gear'],                 () => loadDemo('gears')],
  [['tesseract', 'hypercube'],        () => loadDemo('tesseract')],
  [['sixteen cell', '16 cell'],       () => loadDemo('cell16')],
  [['five cell', '5 cell', 'pentachoron'], () => loadDemo('cell5')],
  // stage controls
  [['present', 'fullscreen', 'full screen'], () => setPresent(true)],
  [['exit', 'controls'],              () => setPresent(false)],
  [['grid'],                          () => $('tGrid').click()],
  [['bloom', 'glow'],                 () => $('tBloom').click()],
  [['share', 'link'],                 () => $('shareBtn').click()],
];
// "delete part 3" / "remove fuel tank": number wins (Web Speech usually returns digits; word fallback for
// spoken numbers, longest-first so "sixteen" doesn't read as "six"), then the longest matching part name.
const NUMWORD = { one:1,two:2,three:3,four:4,five:5,six:6,seven:7,eight:8,nine:9,ten:10,eleven:11,twelve:12,
  thirteen:13,fourteen:14,fifteen:15,sixteen:16,seventeen:17,eighteen:18,nineteen:19,twenty:20 };
function spokenNum(t) {   // whole-word only: "\b" keeps "cone"→"one" and "sixteen"→"six" from firing false numbers
  const d = t.match(/\b\d+\b/); if (d) return +d[0];
  for (const w in NUMWORD) if (new RegExp(`\\b${w}\\b`).test(t)) return NUMWORD[w];
  return null;
}
function voiceDelete(t) {   // returns the label removed, or null if nothing matched (so runVoice can fall through)
  const live = (p) => p && p.visible && !removed.includes(p);
  const n = spokenNum(t);
  if (n != null && live(parts[n - 1])) { sinkPart(parts[n - 1]); return parts[n - 1].userData.label; }
  let best = null, bestLen = 0;
  for (const p of parts) {
    const nm = p.userData.name && p.userData.name.toLowerCase();
    if (nm && nm.length > bestLen && t.includes(nm) && live(p)) { best = p; bestLen = nm.length; }
  }
  if (best) { sinkPart(best); return best.userData.label; }
  return null;
}
// Voice axis-lock: freeze pitch (X), yaw (Y) or roll (Z) hands-free; "unlock" frees them. Web Speech
// mangles single letters, so accept the usual homophones as WHOLE words ("ex/axe", "why", "zee/zed").
function setAxisLock(axis, on) {   // drive the existing panel checkbox so its change listener updates lockX/Y/Z and the UI stays in sync
  const el = $('lock' + axis.toUpperCase()); if (!el) return;
  el.checked = on; el.dispatchEvent(new Event('change'));
}
function runVoice(transcript) {   // returns the matched phrase (for testing) or null
  const t = transcript.toLowerCase();
  if (t.includes('delete') || t.includes('remove')) {   // handled before the keyword list so "part 3" isn't misread
    const hit = voiceDelete(t);
    if (hit) { initAudio(); blip(200, 0.13, 'sawtooth', 0.06); stateEl.textContent = `🗑 Deleted ${hit} — say "restore" to bring it back`; return 'delete'; }
  }
  const ax = axisFromVoice(t);   // "lock x/y/z" freezes that rotation axis, "unlock" frees all — before the keyword list
  if (ax === 'unlock') { for (const a of ['x', 'y', 'z']) setAxisLock(a, false); initAudio(); blip(520, 0.08); stateEl.textContent = '🔓 Axes free — rotate on all axes'; return 'unlock'; }
  if (ax) { setAxisLock(ax, true); initAudio(); blip(300, 0.09, 'square'); stateEl.textContent = `🔒 ${ax.toUpperCase()} axis locked — say "unlock" to free`; return 'lock ' + ax; }
  for (const [keys, fn] of VOICE) if (keys.some((k) => t.includes(k))) {
    fn(); initAudio(); blip(760, 0.05); stateEl.textContent = `♪ heard “${t.trim()}”`;
    return keys[0];
  }
  return null;
}
let recog = null, voiceOn = false;
function setVoice(on) {
  if (!SR) { stateEl.textContent = 'Voice needs Chrome (Web Speech API not found)'; return; }
  voiceOn = on;
  $('voiceBtn').classList.toggle('on', on);
  $('voiceBtn').classList.toggle('listening', on);
  if (on) {
    if (!recog) {
      recog = new SR(); recog.continuous = true; recog.interimResults = false; recog.lang = 'en-US';
      recog.onresult = (e) => { const r = e.results[e.results.length - 1]; if (r.isFinal) runVoice(r[0].transcript); };
      recog.onend = () => { if (voiceOn) { try { recog.start(); } catch {} } };   // continuous drops after a pause; keep it alive while enabled
      recog.onerror = (e) => { if (e.error === 'not-allowed' || e.error === 'service-not-allowed') { setVoice(false); stateEl.textContent = 'Mic blocked — allow it, then click Voice'; } };
    }
    try { recog.start(); } catch {}
    stateEl.textContent = '🎙 Listening — "iron man", "explode", "delete part 3", "lock x", "present"…';
    blip(880, 0.08);
  } else { recog && recog.stop(); blip(320, 0.08); }
}
$('voiceBtn').addEventListener('click', () => { initAudio(); setVoice(!voiceOn); });
window._runVoice = runVoice;   // ponytail: exposed so the mic-less browser pane can test the keyword→action map directly

// ---------- proximity-to-inspect: the nearest part to the hand IN SCREEN SPACE ----------
// No raycast — we project each visible part's centre and the hand's anchor to the screen and take the
// closest part within REACH. Isotropic: the x-gap is scaled by the aspect ratio so the reach zone is a
// round disc on screen, not an ellipse. This reads exactly as it looks and never "hits" removed parts.
const labelEl = $('label');
const _tipW = new THREE.Vector3(), _d = new THREE.Vector3(), _binW = new THREE.Vector3();  // grabbed-part carry tracking (pivot-local) + bin proximity (NDC, screen-space)
const _tipWorld = new THREE.Vector3(), _cw = new THREE.Vector3(), _coffP = new THREE.Vector3(), _rest = new THREE.Vector3();  // absolute grab: pinch point + part-centre offset + rest position, all worked in the part's parent space
const _pw = new THREE.Vector3(), _hp = new THREE.Vector3();                                 // scratch for screen-space projection
let highlighted = null;
// Highlight = re-tint the part's emissive: blue (HL_INFO) while you inspect it, red (HL_GRAB) once it's
// grabbed. The original emissive is cached per material the first time we touch it, so clearing restores
// the holo skin exactly. Guarded on `m.emissive` so a skinned-off model with a flat material won't throw.
function tint(part, hex, intensity) {
  const m = part?.userData.core?.material;
  if (!m || !m.emissive) return;
  if (m.userData.emissive0 == null) m.userData.emissive0 = m.emissive.getHex();
  m.emissive.setHex(hex); m.emissiveIntensity = intensity;
}
function restoreEmissive(part) {
  const m = part?.userData.core?.material;
  if (m && m.emissive && m.userData.emissive0 != null) { m.emissive.setHex(m.userData.emissive0); m.emissiveIntensity = BASE_EMISSIVE; }
}
function setHighlight(part, hex = HL_INFO) {
  if (highlighted && highlighted !== part) restoreEmissive(highlighted);
  highlighted = part;
  tint(part, hex, HIGHLIGHT_EMISSIVE);
}
function clearHighlight() { if (highlighted) restoreEmissive(highlighted); highlighted = null; labelEl.style.opacity = '0'; }
function reachPart(handWorld) {          // nearest visible part to the hand on screen; { part, d } in NDC units, or null
  _hp.set(handWorld.x, handWorld.y, handWorld.z).project(camera);
  const aspect = innerWidth / innerHeight;
  let best = null, bestD = Infinity;
  for (const p of parts) {
    if (!p.visible) continue;
    p.localToWorld(_pw.copy(p.userData.coff)).project(camera);   // the part's geometric centre, not its (arbitrary) origin
    const dx = (_pw.x - _hp.x) * aspect, dy = _pw.y - _hp.y;
    const d = Math.hypot(dx, dy);
    if (d < bestD) { bestD = d; best = p; }
  }
  return best ? { part: best, d: bestD } : null;
}
function inspectPart(part) {              // blue highlight + info card floated at the part's screen position
  if (!part) { clearHighlight(); return; }
  setHighlight(part, HL_INFO);
  part.localToWorld(_pw.copy(part.userData.coff)).project(camera);
  const x = (_pw.x * 0.5 + 0.5) * innerWidth, y = (-_pw.y * 0.5 + 0.5) * innerHeight;
  labelEl.innerHTML = `${part.userData.label}<span class="lsub">${part.userData.info}</span>`;   // name + honest info card, floated at the part
  labelEl.style.transform = `translate(-50%,-150%) translate(${x}px,${y}px)`;
  labelEl.style.opacity = '1';
}

// ---------- measure mode: the model's bounding dimensions as a cage + floating W/H/D labels ----------
// Turns the toy into a tool. NOT a two-point hand pick (unreliable with proximity tracking, and meaningless
// on a single-mesh model): it draws the authored bounding box captured pre-fit in fitToView, so the numbers
// are the model's REAL units (mm for most CAD), not the on-stage fit size, and don't change when you zoom.
// The cage lives in `pivot` (rotates/scales with the model) but carries the model's own fit scale+offset;
// labels project three edge-midpoints to the screen each frame. depthTest off => the cage reads through a
// solid Normal-mode model, like a CAD viewer. ponytail: units are the model's own; add a "set real length"
// calibration only if someone needs absolute mm off a glTF that wasn't authored to scale.
const measureGroup = new THREE.Group(); measureGroup.renderOrder = 999; measureGroup.visible = false;
pivot.add(measureGroup);
const dimEls = { x: $('dimX'), y: $('dimY'), z: $('dimZ') };
const _mv = new THREE.Vector3(), _dcen = new THREE.Vector3();
let measureOn = false, _dims = null;
const fmtDim = (n) => n >= 100 ? n.toFixed(0) : n >= 10 ? n.toFixed(1) : n.toFixed(2);  // legible from mm CAD to unit-scale glTF
function buildMeasure() {
  for (const c of [...measureGroup.children]) { measureGroup.remove(c); c.geometry.dispose(); c.material.dispose(); }
  _dims = model.userData.dims; if (!_dims) return;                    // set in fitToView for every model incl. the rocket
  _dcen.copy(model.userData.dcen);
  const line = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(_dims.x, _dims.y, _dims.z)),
    new THREE.LineBasicMaterial({ color: 0x35c8ff, transparent: true, opacity: 0.5, depthTest: false }));
  line.position.copy(_dcen); measureGroup.add(line);
  measureGroup.scale.copy(model.scale); measureGroup.position.copy(model.position);  // ride the fit transform; pivot adds rotate/zoom
}
function placeDim(el, lx, ly, lz, text) {                             // one box point (authored-local) -> screen px
  measureGroup.localToWorld(_mv.set(lx, ly, lz)).project(camera);
  if (_mv.z > 1) { el.style.opacity = '0'; return; }                 // behind the camera
  const x = (_mv.x * 0.5 + 0.5) * innerWidth, y = (-_mv.y * 0.5 + 0.5) * innerHeight;
  el.textContent = text; el.style.transform = `translate(-50%,-50%) translate(${x}px,${y}px)`; el.style.opacity = '1';
}
function updateMeasure() {
  if (!measureOn || !_dims) return;
  const hx = _dims.x / 2, hy = _dims.y / 2, hz = _dims.z / 2, c = _dcen;
  placeDim(dimEls.x, c.x, c.y - hy, c.z + hz, 'W ' + fmtDim(_dims.x));  // bottom-front edge
  placeDim(dimEls.y, c.x - hx, c.y, c.z + hz, 'H ' + fmtDim(_dims.y));  // front-left vertical edge
  placeDim(dimEls.z, c.x + hx, c.y - hy, c.z, 'D ' + fmtDim(_dims.z));  // bottom-right edge
}
function setMeasure(on) {
  measureOn = on; measureGroup.visible = on;
  if (on) buildMeasure(); else for (const k in dimEls) dimEls[k].style.opacity = '0';
}
$('tDims').addEventListener('change', (e) => { setMeasure(e.target.checked); blip(640, 0.05); });

// ---------- part labels: every part names itself as the model comes apart ----------
// The teaching payoff. Pull the model apart and each piece floats its own name, so an exploded view
// reads like a labelled diagram — the rocket's "Payload Fairing / Fuel Tank / Engine Bell", or whatever
// an uploaded model's parts are called. No toggle: names fade in with the explode amount and out as it
// closes. One pooled DOM chip per part (rebuilt on every load), reprojected to screen each frame — same
// project()->NDC->pixels idiom the Measure labels use. buildPartLabels runs during the first collectParts
// (init), so it resolves its container locally and writes the module-scoped partLabelEls declared up top.
const _pl = new THREE.Vector3();
function buildPartLabels() {
  const wrap = $('partLabels');
  wrap.textContent = '';                                             // drop the previous model's chips
  partLabelEls = parts.map((p) => {
    const el = document.createElement('div');
    el.className = 'plabel'; el.textContent = p.userData.label || p.name || 'Part';
    wrap.appendChild(el); return el;
  });
}
function updatePartLabels() {
  const fade = clamp((current.explode - 0.1) / 0.3, 0, 1);           // hidden when assembled, ramps in as parts separate
  for (let i = 0; i < parts.length; i++) {
    const el = partLabelEls[i], p = parts[i];
    if (!el) continue;
    if (fade <= 0 || removed.includes(p) || !p.visible) { el.style.opacity = '0'; continue; }
    p.localToWorld(_pl.copy(p.userData.coff)).project(camera);       // part's own centre -> screen
    if (_pl.z > 1) { el.style.opacity = '0'; continue; }             // behind the camera
    const x = (_pl.x * 0.5 + 0.5) * innerWidth, y = (-_pl.y * 0.5 + 0.5) * innerHeight;
    el.style.transform = `translate(-50%,-50%) translate(${x}px,${y}px)`;
    el.style.opacity = String(fade);
  }
}

// ---------- part removal: strip the hologram down a component at a time ----------
// Materials are dimmed rather than parts hidden, so "isolating" a locked part reads as the rest
// ghosting out — the Iron Man 1 shot. Original opacity is cached per material on first touch.
// ponytail: a model whose parts SHARE one material ghosts them together; holo skin (on by default)
// clones a material per mesh, so this only shows with the skin off. Per-part material clones if so.
function setPartOpacity(part, f) {
  part.traverse(o => {
    if (!o.isMesh) return;
    for (const m of [o.material].flat()) {
      if (m.userData.op0 == null) m.userData.op0 = m.opacity;
      m.transparent = true;
      m.opacity = m.userData.op0 * f;
    }
  });
}
function ghostOthers(keep) { for (const p of parts) setPartOpacity(p, keep && p !== keep ? GHOST_FACTOR : 1); }
function sinkPart(p) {                       // dropped in the dustbin: dissolve in place, book it as removed
  if (!p || removed.includes(p)) return;
  ejecting.push({ p, t: 0, pull0: p.userData.pull, sink: true });
  blip(150, 0.2, 'sawtooth', 0.06); setTimeout(() => blip(80, 0.28, 'sine', 0.05), 60);
}
function restoreParts() {                   // put the whole assembly back (nothing is destroyed, only hidden)
  for (const p of [...removed, ...ejecting.map(e => e.p)]) { p.visible = true; p.userData.pull = p.userData.pullT = 0; setPartOpacity(p, 1); }
  removed.length = 0; ejecting.length = 0;
  aim = AIM_OFF; locked = null; grabTip = null; binShow = false; clearHighlight();
  updateStats();
}
$('restore').addEventListener('click', () => { initAudio(); blip(560, 0.1, 'triangle'); setTimeout(() => blip(840, 0.12, 'triangle'), 80); restoreParts(); });

// ---------- MediaPipe hand tracking ----------
let handLandmarker = null, drawUtils = null, running = false, lastVideoTime = -1;
let prevCenter = null, prevSpread = null, prevAngle = null, idle = true;
let aim = AIM_OFF, locked = null, grabTip = null;   // extraction machine: state, the grabbed part, and the palm pos (pivot-local) where the grab began
let thumbLatch = false;               // edge-latch so a held 👍 fires reset once, not every frame
let openHold = 0;                     // consecutive open-hand frames while carrying — debounces release so a mid-drag pinch wobble doesn't drop/bin the part
let dwellId = null, dwellStart = 0;   // point-to-inspect dwell: which part id, and when the point settled on it (info card commits after DWELL_MS)
let thumbStart = 0;                   // when the current 👍 hold began — reset only fires after THUMB_HOLD_MS
let dragging = false;                 // true while a pinch is actively rotating the model
const spinVel = { rx: 0, ry: 0 };     // leftover angular velocity after you let go (flick-to-spin)

async function initHands() {
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  const opts = { baseOptions: { modelAssetPath: MODEL_URL }, runningMode: 'VIDEO', numHands: 2 };
  try { // GPU is faster but unavailable on some machines — fall back to CPU
    handLandmarker = await HandLandmarker.createFromOptions(vision, { ...opts, baseOptions: { ...opts.baseOptions, delegate: 'GPU' } });
  } catch { handLandmarker = await HandLandmarker.createFromOptions(vision, opts); }
  drawUtils = new DrawingUtils(overlay.getContext('2d'));
}

let starting = false;
async function startCamera() {
  if (running || starting) return;   // a second click would open a second camera stream and a second tracker
  starting = true;
  try {
    errEl.textContent = ''; initAudio();
    if (!handLandmarker) { stateEl.textContent = 'Loading model…'; await initHands(); }
    video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    await video.play();
    overlay.width = video.videoWidth; overlay.height = video.videoHeight;
    running = true; stateEl.textContent = 'Show a hand';
    if (recEl) { recEl.textContent = 'LIVE'; recEl.classList.add('on'); }
    const boot = $('boot');
    if (boot) { $('bootsub').textContent = 'HAND TRACKING ONLINE'; blip(600, 0.12, 'triangle'); setTimeout(() => boot.classList.add('hidden'), 900); }
  } catch (err) {
    errEl.textContent = 'Camera/model error: ' + err;
    $('boot')?.classList.add('hidden');   // otherwise the intro card covers the scene forever and the app looks dead
  } finally { starting = false; }
}
$('start').addEventListener('click', startCamera);

function applyGestures(hands) {
  // Gesture vocabulary (proximity model): ONE hand drives PARTS — bring it NEAR a part to inspect it,
  // PINCH to grab & carry, OPEN to release (over 🗑 = remove, else snap home), ✊ FIST + move = rotate the
  // whole model, 👍 = reset. TWO COMMITTED hands drive the WHOLE MODEL: 🖐🖐 = zoom, ✊✊ = explode + twist-roll.
  // OPEN is neutral — an idle hand does nothing. Every hand is live; we classify each once and act.
  const classified = hands.map(h => ({ h, pose: handPose(h) }));
  dragging = false;                                                        // set true only while rotating
  poseEl.textContent = classified.map((a, i) => `H${i + 1} ${a.pose} ${pinchStrength(a.h).toFixed(2)}`).join('    ');

  // Two hands only enter WHOLE-MODEL mode when they COMMIT to the same pose (both open => zoom, both fist =>
  // explode). Otherwise a resting second hand mustn't hijack a solo pinch/fist, so we pick the dominant hand
  // by pose rank — the one actually doing something wins.
  const twoHand = classified.length >= 2;
  const bothOpen = twoHand && classified[0].pose === 'open' && classified[1].pose === 'open';
  const bothFist = twoHand && classified[0].pose === 'fist' && classified[1].pose === 'fist';
  const modelMode = bothOpen || bothFist;
  const solo = classified.length === 1 ? classified[0]
             : (twoHand && !modelMode ? classified.reduce((a, b) => RANK[b.pose] > RANK[a.pose] ? b : a) : null);
  const soloC = solo ? handCenter(solo.h) : null;
  const soloPinch = solo ? pinchPoint(solo.h) : null;                       // thumb/index midpoint: where the hand actually pinches a part
  if (solo && solo.pose === 'thumbsup') { if (!thumbStart) thumbStart = performance.now(); }  // 👍 up: begin/keep the hold timer
  else { thumbLatch = false; thumbStart = 0; }                                                 // thumb dropped: re-arm reset and clear the hold timer

  // ---- Section takes the hand outright while it's armed, before any other gesture can claim it:
  // a cut plane you have to share with rotate-on-pinch would swing the model every time you turned
  // your wrist. Make a fist (or drop your hand) and the plane freezes where you left it, so you can
  // take your hand out of frame and actually look at the cross-section.
  if (sectionOn) {
    if (solo) updateCut(solo.h);
    prevCenter = null; prevSpread = null; prevAngle = null; idle = false; clearHighlight();
    stateEl.textContent = solo ? '◧ Sectioning — turn your palm to cut · ✊ to freeze' : '◧ Section frozen — show a hand to move it';
    setMode('section'); updateStats(hands.length);
    return;
  }

  // ---- Part handling, stepped every frame BEFORE anything else can claim the hand. Bring the hand NEAR a
  // part (screen-space proximity, not a raycast) to INSPECT it; PINCH while it's in reach to GRAB it onto
  // your hand, then move — the part rides the hand 1:1. OPEN to let go: over the bin it's removed, else it
  // snaps home. No timer and no proximity delete, so carrying a part through the bin never deletes it — only
  // the deliberate open-over-bin does. We anchor selection AND carry to the PINCH POINT (thumb/index
  // midpoint) — exactly where your fingers close on the part on screen. The palm centroid sat back toward
  // the wrist, so pinching at a part measured from the wrong spot and grabbed the wrong thing (the bug).
  const anchorLm = solo ? (solo.pose === 'point' ? solo.h[INDEX_TIP] : soloPinch) : null;  // point => the fingertip you aim WITH; pinch => the thumb/index midpoint where the fingers close on a part
  const tip = anchorLm ? landmarkToWorld(anchorLm, HAND_SPAN, HAND_DEPTH) : null;
  if (tip) { _tipW.set(tip.x, tip.y, tip.z); pivot.worldToLocal(_tipW); }     // pinch point in the model's own space
  const reach = tip && aim.phase !== 'grab' ? reachPart(tip) : null;          // nearest part on screen while we're not already carrying
  const reachable = reach && reach.d < REACH ? reach.part : null;             // ...only if it's actually within arm's reach
  let inBin = false;
  if (aim.phase === 'grab' && locked && tip && grabTip) {                     // grabbed: the part's CENTRE rides the pinch point
    // Glue the part's geometric centre to the pinch point (thumb/index midpoint) in ABSOLUTE terms —
    // "wherever my fingertips go, the middle of the body goes." The old code added a relative hand-delta
    // measured in PIVOT space onto a part whose .position lives in its (often nested, differently-scaled)
    // PARENT space, so a GLB part barely moved or shot off, and it never actually reached the fingertip —
    // it sat at its exploded rest spot. Now solve, in the part's own parent space, for the pull offset that
    // lands the centre exactly on the fingertip, then ease that WHOLE vector (no teleport on a turn toward
    // the bin). Routed through the existing home + dir*explode + pdir*pull path so drop-eases-home and the
    // bin test are unchanged.
    const par = locked.parent;
    _cw.copy(locked.userData.coff); locked.localToWorld(_cw); par.worldToLocal(_cw);   // part centre: local -> world -> parent space
    _coffP.copy(_cw).sub(locked.position);                                   // where the centre sits relative to the part's origin, in parent space
    _tipWorld.set(tip.x, tip.y, tip.z); par.worldToLocal(_tipWorld);         // pinch point -> parent space
    _rest.copy(locked.userData.home).addScaledVector(locked.userData.dir, current.explode * EXPLODE_K);  // the part's current resting spot (home + live explode)
    _d.copy(_tipWorld).sub(_coffP).sub(_rest);                               // pull offset from rest that puts the centre on the fingertip
    const cv = locked.userData.carry.lerp(_d, GRAB_SMOOTH), len = cv.length();
    if (len > 1e-6) { locked.userData.pdir.copy(cv).multiplyScalar(1 / len); locked.userData.pull = locked.userData.pullT = len; }
    else { locked.userData.pull = locked.userData.pullT = 0; }               // centre already on the fingertip
    // Bin hit is SCREEN-space, not 3D: MediaPipe's hand z is noisy/compressed, so a world distance to the
    // bin rarely closes on the bin's fixed z. Project the part's centre AND the bin's MOUTH to NDC and compare
    // in 2D (aspect-corrected) — depth stops mattering, like dropping a file on a desktop trash icon. Anchor to
    // the MOUTH (rim, +0.41 above the bin origin), not the body centre: you drop a part ONTO the opening, and a
    // tall part (a long rod) hovers there with its own centre well above the bin centre — testing against the
    // centre meant its middle never got close enough, so it read as "always above, never in."
    locked.localToWorld(_pw.copy(locked.userData.coff)).project(camera);      // carried part's centre, on screen
    _binW.set(bin.position.x, bin.position.y + 0.41, bin.position.z).project(camera);  // the bin's MOUTH, on screen
    const aspect = innerWidth / innerHeight;
    inBin = Math.hypot((_pw.x - _binW.x) * aspect, _pw.y - _binW.y) < BIN_HIT;
  }
  binArmed = inBin;                                                           // drives the bin's flare in the render loop
  // Debounce release: while carrying, a lone 'open' frame (the pinch loosening as your hand moves) must NOT
  // drop or bin the part — only a SUSTAINED open does. Mask a brief open as 'hold' so aimStep keeps carrying.
  let effPose = solo?.pose ?? null;
  if (aim.phase === 'grab') {
    if (effPose === 'open') { openHold++; if (openHold < RELEASE_FRAMES) effPose = 'hold'; }
    else openHold = 0;
  }
  aim = aimStep(aim, { pose: effPose, id: reachable?.userData.pid ?? null, present: !!solo, inBin });
  if (aim.phase !== 'aim') dwellId = null;                                  // left the inspect: next point restarts the ~1.5s dwell from zero
  if (aim.action === 'grab') { locked = parts[aim.id]; locked.userData.carry.set(0, 0, 0); setHighlight(locked, HL_GRAB); ghostOthers(locked); grabTip = _tipW.clone(); showBin(true); openHold = 0; }
  else if (aim.action) {                                                     // 'remove' (binned) or 'drop' (let go)
    if (aim.action === 'remove') sinkPart(locked);
    else if (locked) locked.userData.pullT = 0;                            // dropped: eases back home
    locked = null; grabTip = null; ghostOthers(null); showBin(false); clearHighlight();
  }

  if (modelMode) {                                                          // two COMMITTED hands => whole-model mode
    const a0 = classified[0].h, a1 = classified[1].h;
    const spread = twoHandSpread(a0, a1);
    const ang = twoHandAngle(a0, a1);                                       // twist both hands (like a wheel) => roll
    // Roll ONLY while exploding (two fists): zoom (two palms) is a pure scale, so the model won't drift
    // in orientation while you resize it. The deadzone drops sub-threshold wobble so only a deliberate
    // twist rolls; passing zooming=!bothFist suppresses roll during a two-palm zoom.
    target.rz += rollDelta(prevAngle, ang, { zooming: !bothFist, deadzone: TWIST_DEADZONE, mirror: MIRROR_X });
    prevAngle = ang;                                                        // tracked even while zooming, so no jump on release
    prevCenter = null; idle = false; clearHighlight();
    if (bothFist) {                                                         // two fists => explode INCREMENTALLY: widen the gap ADDS, narrow REMOVES, release HOLDS — so it persists like zoom
      if (prevSpread != null) target.explode = clamp(target.explode + (spread - prevSpread) * EXPLODE_SPEED, 0, 1);
      prevSpread = spread;
      stateEl.textContent = target.explode > 0.05 ? `✊✊ Exploded ${Math.round(target.explode * 100)}% · twist to roll` : '✊✊ Pull fists apart to explode · twist to roll';
      setMode('explode');
    } else {                                                                // two palms => pure zoom
      if (prevSpread != null) target.scale = clamp(target.scale + (spread - prevSpread) * ZOOM_SPEED, SCALE_MIN, SCALE_MAX);
      prevSpread = spread;
      stateEl.textContent = '🖐🖐 Zoom — move hands apart / together'; setMode('zoom');
    }
  } else if (aim.phase === 'grab') {                                        // a pinch is holding a part; it rides the hand 1:1
    prevCenter = null; prevSpread = null; prevAngle = null; idle = false;
    setHighlight(locked, HL_GRAB); labelEl.style.opacity = '0';             // the info card was for inspecting; the HUD line drives the grab
    stateEl.textContent = inBin
      ? `🗑 ${locked?.userData.label} over the bin — 🖐 open to remove`
      : `🤏 Holding ${locked?.userData.label} — 🖐 open to drop · carry to 🗑 to remove`;
    setMode('grab');
  } else if (aim.phase === 'aim') {                                        // POINT at a part: identify it after a short dwell (a passing glance shouldn't fire)
    prevCenter = null; prevSpread = null; prevAngle = null; idle = false;
    const part = parts[aim.id];
    if (dwellId !== aim.id) { dwellId = aim.id; dwellStart = performance.now(); }  // moved to a new part: restart the dwell timer
    const held = performance.now() - dwellStart;
    if (held >= DWELL_MS) {                                                 // dwell complete => commit the info card + full label
      inspectPart(part);
      stateEl.textContent = `ℹ ${part?.userData.label} — 🤏 pinch to grab · say “delete part ${part?.userData.num}”`;
    } else {                                                                // still settling => blue glow + progress, no card yet
      setHighlight(part, HL_INFO); labelEl.style.opacity = '0';
      stateEl.textContent = `☝ Identifying ${part?.userData.label}… ${Math.round(held / DWELL_MS * 100)}%`;
    }
    setMode('inspect');
  } else if (solo && solo.pose === 'fist') {                                // one fist + move => rotate the whole model
    const c = soloC;
    if (prevCenter) {
      const dry = (c.x - prevCenter.x) * ROT_SPEED * MIRROR_X;
      const drx = (c.y - prevCenter.y) * ROT_SPEED;
      target.ry += dry; target.rx += drx;
      spinVel.ry = spinVel.ry * (1 - SPIN_CAPTURE) + dry * SPIN_CAPTURE;    // recent-weighted flick speed
      spinVel.rx = spinVel.rx * (1 - SPIN_CAPTURE) + drx * SPIN_CAPTURE;
    }
    prevCenter = c; prevSpread = null; prevAngle = null; idle = false; dragging = true; clearHighlight();
    stateEl.textContent = '✊ Rotate — move your fist to turn the model'; setMode('rotate');
  } else if (solo && solo.pose === 'thumbsup') {                           // 👍 (one hand, HELD) => reset everything — the hold stops a fist misread from wiping your work
    prevCenter = null; prevSpread = null; prevAngle = null; idle = false; clearHighlight();
    const held = performance.now() - thumbStart;
    if (classified.length === 1 && held >= THUMB_HOLD_MS) {
      if (!thumbLatch) { thumbLatch = true; $('reset').click(); }           // fires once per hold
      stateEl.textContent = '👍 Reset';
    } else {
      stateEl.textContent = `👍 Hold to reset… ${Math.round(clamp(held / THUMB_HOLD_MS, 0, 1) * 100)}%`;
    }
    setMode('idle');
  } else {                                                                   // nothing armed => drift
    prevCenter = null; prevSpread = null; prevAngle = null; idle = true; clearHighlight();
    stateEl.textContent = hands.length
      ? '☝ Point at a part to identify'
      : 'Show a hand';
    setMode('idle');
  }
  updateStats(hands.length);
}

function drawHands(hands) {
  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  for (const lm of hands) {
    drawUtils.drawConnectors(lm, HandLandmarker.HAND_CONNECTIONS, { color: '#7CFFB2', lineWidth: 3 });
    drawUtils.drawLandmarks(lm, { color: '#8ab4ff', radius: 3 });
  }
}

function loop() {
  requestAnimationFrame(loop);
  if (running && handLandmarker && video.currentTime !== lastVideoTime && video.videoWidth) {
    lastVideoTime = video.currentTime;
    const hands = handLandmarker.detectForVideo(video, performance.now()).landmarks || [];
    if (hands.length) lastHandTs = performance.now();
    drawHands(hands);
    applyGestures(hands);
    updateHandViz(hands);
  }
  // recede to the live cockpit strip when a hand has driven recently and the mouse has gone idle;
  // grab the mouse (or drop your hand) and the full control panel glides back.
  const t = performance.now();
  hud.classList.toggle('cockpit', running && !presenting && (t - lastHandTs < 700) && (t - lastMouseTs > 2000));
  if (!dragging && (spinVel.rx || spinVel.ry)) {   // flick momentum: keep spinning, then ease to rest
    target.ry += spinVel.ry; target.rx += spinVel.rx;
    spinVel.ry *= SPIN_FRICTION; spinVel.rx *= SPIN_FRICTION;
    if (Math.hypot(spinVel.rx, spinVel.ry) < SPIN_MIN) spinVel.rx = spinVel.ry = 0;
  } else if (idle && autoSpin && performance.now() > manualT) target.ry += IDLE_SPIN;   // gentle turntable so the hologram feels alive
  if (playing) { playT += 0.008; target.explode = 0.5 - 0.5 * Math.cos(playT); }   // attract loop: breathe the explode in and out
  if (lockX) target.rx = current.rx;   // held axes stop accumulating, so unlocking resumes smoothly (no snap)
  if (lockY) target.ry = current.ry;
  if (lockZ) target.rz = current.rz;
  current.rx += (target.rx - current.rx) * SMOOTH; // ease toward target every frame
  current.ry += (target.ry - current.ry) * SMOOTH;
  current.rz += (target.rz - current.rz) * SMOOTH;
  current.scale += (target.scale - current.scale) * SMOOTH;
  current.explode += (target.explode - current.explode) * SMOOTH;
  if (model.userData.tick) model.userData.tick();   // 4D polytopes redraw their projected wireframe each frame; explode folds them deeper in 4D
  // Ejecting parts keep accelerating outward while they fade, so a removal reads as the piece being
  // thrown clear of the assembly rather than just blinking off. `pull` eases like everything else,
  // which is what makes a hand-pull feel weighted and a released part settle back instead of snapping.
  for (let i = ejecting.length - 1; i >= 0; i--) {
    const e = ejecting[i];
    e.t += EJECT_SPEED;
    if (!e.sink) e.p.userData.pullT = e.pull0 + e.t * 1.5;   // bin removals dissolve in place; this is the legacy fly-out
    setPartOpacity(e.p, Math.max(0, 1 - e.t));
    if (e.t >= 1) { e.p.visible = false; e.p.userData.pull = e.p.userData.pullT = 0; ejecting.splice(i, 1); removed.push(e.p); updateStats(); }
  }
  for (const p of parts) {
    p.userData.pull += (p.userData.pullT - p.userData.pull) * SMOOTH;
    p.position.copy(p.userData.home)
      .addScaledVector(p.userData.dir, current.explode * EXPLODE_K)
      .addScaledVector(p.userData.pdir, p.userData.pull);
  }
  pivot.rotation.set(current.rx, current.ry, current.rz);
  pivot.scale.setScalar(current.scale);
  for (const r of reticle.children) r.rotateOnWorldAxis(_up, 0.004 * r.userData.dir); // flat spin, projector look
  if (binShow || bin.visible) {   // dustbin eases in while a part is grabbed, out otherwise; slow spin so it reads as live
    const [skin, wireMat] = bin.userData.mats;
    skin.opacity += ((binShow ? 0.9 : 0) - skin.opacity) * 0.15;
    // Flare when a part is over the mouth: brighter glow + a small pulse so you SEE "let go now to delete."
    // Without this the drop zone was invisible and a hovering part just looked stuck above the bin.
    skin.emissiveIntensity += ((binArmed ? 2.2 : 0.6) - skin.emissiveIntensity) * 0.2;
    const pulse = binArmed ? 1.12 + Math.sin(performance.now() * 0.012) * 0.06 : 1;
    bin.scale.setScalar(bin.scale.x + (pulse - bin.scale.x) * 0.25);
    wireMat.opacity = skin.opacity * (binArmed ? 1 : 0.55); bin.visible = skin.opacity > 0.02; bin.rotation.y += binArmed ? 0.04 : 0.012;
  }
  updateMeasure();   // reproject the W/H/D labels onto the (possibly rotated/zoomed) cage
  updatePartLabels(); // float each part's name while the model is exploded
  updateStats();     // refresh the HUD readout every frame — keyboard/voice/auto-play/share change current.explode & scale even with the camera off
  composer.render();
}
updateStats(0);   // seed the HUD readout (rocket = 7 parts) before the first frame

// Restore a shared view from the URL hash, if any (see Share view). loadDemo→swapModel resets the
// transform, so apply the saved transform AFTER it, not before.
function applyHash() {
  if (location.hash.length < 2) return;
  const p = new URLSearchParams(location.hash.slice(1));
  const applyView = () => {                 // rotation / scale / explode / mode — must run AFTER any swapModel (which resets them)
    const num = (k, d) => { const v = parseFloat(p.get(k)); return Number.isFinite(v) ? v : d; };
    target.rx = current.rx = num('rx', 0); target.ry = current.ry = num('ry', 0); target.rz = current.rz = num('rz', 0);
    target.scale = current.scale = clamp(num('s', 1), SCALE_MIN, SCALE_MAX);
    target.explode = current.explode = clamp(num('e', 0), 0, 1);
    const mode = p.get('mode'); if (RENDER_MODES.includes(mode)) setRenderMode(mode);
  };
  const m = p.get('m');
  if (m && (MODELS[m] || FILE_DEMOS[m])) loadDemo(m, applyView);   // procedural or GLB demo; applyView fires once it's in place
  else applyView();                                                // no model in the link (or an uploaded file): apply to whatever's loaded
}
applyHash();
loop();

// The pre-flight airlock (enter.html) hands us the user's camera/mic choices via sessionStorage. Permission
// was already granted there, so these resolve without a second prompt — the lab boots the way they armed it.
// Read once, then clear: reloading app.html directly shouldn't force the devices back on.
try {
  if (sessionStorage.getItem('origin.cam') === '1') startCamera();
  if (sessionStorage.getItem('origin.voice') === '1') { initAudio(); setVoice(true); }
  sessionStorage.removeItem('origin.cam'); sessionStorage.removeItem('origin.voice');
} catch {}
