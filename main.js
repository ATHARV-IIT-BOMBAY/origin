// HoloControl — rotate & zoom a 3D model with webcam hand gestures.
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
import { handCenter, pinchStrength, twoHandSpread, twoHandAngle, rollDelta, landmarkToWorld, handPose, fitTransform, aimStep, AIM_OFF, palmPlane, INDEX_TIP } from './gestures.js';

// ---- Tuning knobs. A webcam is a messy sensor; these are the calibration dials. ----
// The first four are `let` because the on-screen calibration panel adjusts them live.
let ROT_SPEED = 6.0;      // how far a hand move rotates the model
let ZOOM_SPEED = 6.0;     // how strongly two-hand spread scales the model
let SMOOTH = 0.20;        // 0..1 low-pass ease; lower = smoother but laggier
let MIRROR_X = -1;        // flip so moving your hand right rotates the model right
const SCALE_MIN = 0.3, SCALE_MAX = 4.0;
const HAND_SPAN = 4.2;    // how wide the tracked hand maps into the 3D scene
const HAND_DEPTH = 1.5;   // how strongly landmark depth pushes hand joints in/out
const IDLE_SPIN = 0.0015; // lazy auto-rotate (rad/frame) when you're not controlling it
// Flick-to-spin: while pinch-dragging we track the swipe velocity; on release the model keeps
// spinning and eases to rest (friction). CAPTURE = how much the latest frame feeds the estimate.
const SPIN_FRICTION = 0.96, SPIN_CAPTURE = 0.5, SPIN_MIN = 0.001;
const EXPLODE_K = 1.6;                        // full-explosion expansion: parts push out from center
const EXPLODE_MIN = 0.20, EXPLODE_MAX = 0.75; // two-hand spread range mapped onto 0..1 explosion
const TWIST_DEADZONE = 0.012;                 // rad/frame of two-hand twist to ignore as jitter, so a zoom/hold doesn't drift into roll
const BASE_EMISSIVE = 0.6, HIGHLIGHT_EMISSIVE = 2.4; // part glow: resting vs. aimed-at
// Part removal, pinch-free. Point at a part and HOLD: for GRAB_DWELL_MS you're inspecting it (info card,
// blue glow); hold past that and it grabs onto the fingertip (turns red) and follows your hand 1:1 in the
// model's own space. No pinch, and the grab survives the tracker's pose flickering as you move (that
// flicker dropping the grab was the "it won't move" bug). Removal is the DUSTBIN: drag the grabbed part
// into the bin and it dissolves; let go anywhere else (hand gone / fist / two hands) and it snaps home.
const GRAB_DWELL_MS = 5000;         // ponytail: the 5s "inspect then grab" hold the user asked for; tune here if it feels long
const HL_INFO = 0x2ad0ff, HL_GRAB = 0xff2a2a;  // emissive tint: blue while inspecting, red once it's grabbed and moveable
const BIN_RADIUS = 0.75;            // ponytail: world-space reach to drop a part into the bin — tune with the bin position on real hardware
const EJECT_SPEED = 0.055;  // dissolve speed per frame (~0.3s at 60fps)
const GHOST_FACTOR = 0.16;  // the rest of the model dims to this fraction of its opacity while a part is grabbed
const CUT_SMOOTH = 0.15;    // ease on the section plane — MediaPipe's landmark depth is noisy, and an unsmoothed plane strobes
const CUT_PARKED = 1e6;     // plane constant that puts the cut so far away nothing is clipped (see clipPlane)
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm';

const $ = (id) => document.getElementById(id);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const bar = (f) => '▓'.repeat(Math.round(clamp(f, 0, 1) * 8)).padEnd(8, '░');   // chunky HUD progress bar
const video = $('video'), overlay = $('overlay'), stateEl = $('state'), errEl = $('err'), poseEl = $('pose');
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
renderer.localClippingEnabled = true;   // per-material clipping, so the section plane cuts the model but not its own indicator
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0f);
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
camera.position.set(0, 0, 4);

const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture; // soft PBR reflections
scene.add(new THREE.HemisphereLight(0xffffff, 0x223344, 0.6));
const key = new THREE.DirectionalLight(0xffffff, 2.0); key.position.set(3, 4, 5); scene.add(key);

// Post-processing: bloom makes the emissive wireframe model and the hand joints glow,
// which is what turns a plain mesh into a "hologram". Threshold keeps dim things un-bloomed.
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 0.9, 0.5, 0.7);
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
// (faded out) until a part is grabbed, then it rises in beside the stage; drag the grabbed part within
// BIN_RADIUS of it and the part dissolves. An open can (body + rim + base) in a red holo skin, with a
// wire overlay to match the model's look. Never intercepts the aim ray (all meshes raycast to nothing).
const bin = new THREE.Group(); bin.visible = false; scene.add(bin);
let binShow = false;   // target visibility; the loop eases the bin's opacity toward it
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
bin.position.set(1.7, FLOOR_Y + 0.45, 0.9);   // ponytail: front-right of the stage, within a hand's drag reach — tune with BIN_RADIUS
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
  core.userData.holoNative = true; wire.userData.holoWire = true;  // so applyHoloSkin leaves the rocket alone
  const g = new THREE.Group(); g.add(core, wire);
  g.name = name; g.userData = { label, core };
  return g;
}

// Default model: a multi-part rocket (nose, tanks, engine, fins). It's recognizable, and its
// separately-named parts are exactly what the point-to-inspect and explode gestures act on.
function makeRocket() {
  const R = new THREE.Group(), r = 0.42;
  const nose = holoPart(new THREE.ConeGeometry(r, 0.9, 32), 'nose', 'Nose Cone', 0x143a5a, 0x22c0ff);
  nose.position.y = 1.15;
  const tankA = holoPart(new THREE.CylinderGeometry(r, r, 0.9, 32), 'tankA', 'Fuel Tank');
  tankA.position.y = 0.35;
  const tankB = holoPart(new THREE.CylinderGeometry(r, r, 0.9, 32), 'tankB', 'Oxidizer Tank');
  tankB.position.y = -0.55;
  const engine = holoPart(new THREE.CylinderGeometry(r * 0.55, r, 0.5, 32), 'engine', 'Engine Bell', 0x3a1e08, 0xff7a1a);
  engine.position.y = -1.2;
  R.add(nose, tankA, tankB, engine);
  for (let i = 0; i < 3; i++) {                        // three stabilizer fins around the base
    const fin = holoPart(new THREE.BoxGeometry(0.06, 0.5, 0.42), 'fin' + i, 'Stabilizer Fin');
    const a = (i / 3) * Math.PI * 2;
    fin.position.set(Math.cos(a) * (r + 0.16), -0.95, Math.sin(a) * (r + 0.16));
    fin.rotation.y = -a;
    R.add(fin);
  }
  return R;
}
let model = makeRocket();
pivot.add(model);

function fitToView(obj) { // drop any model centered on the stage with its feet on the floor
  obj.position.set(0, 0, 0); obj.scale.setScalar(1); obj.rotation.set(0, 0, 0);
  obj.updateMatrixWorld(true);                       // measure the raw geometry extent, transform reset
  const box = new THREE.Box3().setFromObject(obj);
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
    p.userData.home = p.position.clone();
    const c = new THREE.Box3().setFromObject(p).getCenter(new THREE.Vector3());
    p.userData.dir = p.parent.worldToLocal(c).sub(p.parent.worldToLocal(mid.clone()));
    // Pulling a part out needs somewhere to pull it, and a dead-center part (a gearbox housing,
    // say) has a zero explode direction by design. Give those one toward the viewer so they can
    // still be extracted, without making them drift during a normal explode.
    p.userData.pdir = p.userData.dir.lengthSq() > 1e-8 ? p.userData.dir : new THREE.Vector3(0, 0, span * 0.35);
    if (!p.userData.core) p.userData.core = firstMesh(p);
    if (p.userData.label == null) p.userData.label = p.name || 'Part';
    // "What it does" for an auto-detected mesh: the mesh data carries no function, only a name and a
    // position in the assembly. So the honest info card is the part's own name (often meaningful —
    // "wheel", "piston" — for a well-authored model) plus which component it is. ponytail: no fake specs.
    p.userData.info = `component ${i + 1} of ${parts.length}`;
  });
  applyClip(root);   // every material has to carry the section plane, or half the model ignores the cut
}
collectParts(model);

// ---------- holographic skin: re-dress ANY uploaded model in the JARVIS look ----------
// Uploaded models arrive with their own materials; this overrides every mesh with a translucent
// emissive-blue material + a cyan wireframe overlay so it reads as a hologram like the built-in
// rocket. Originals are cached so the toggle can restore them. The procedural rocket is marked
// holoNative/holoWire and skipped (it's already holo). Default on — that's the whole aesthetic.
let holoSkin = true, autoSpin = true;
let lockX = false, lockY = false, lockZ = false; // freeze a rotation axis so the model spins cleanly around the free one(s)
let offsetX = 0; // slide the model sideways off the projector center (scene units), set by the Move X slider
function applyHoloSkin(root, on) {
  root.traverse(o => {
    if (!o.isMesh || o.userData.holoNative || o.userData.holoWire) return;
    if (on) {
      if (!o.userData.origMat) o.userData.origMat = o.material;
      if (!o.userData.holoMat) o.userData.holoMat = new THREE.MeshStandardMaterial({
        color: 0x0a2a4a, emissive: 0x0aa0ff, emissiveIntensity: BASE_EMISSIVE,
        metalness: 0.3, roughness: 0.35, transparent: true, opacity: 0.82 });
      if (!o.userData.wire) {
        const w = new THREE.Mesh(o.geometry, new THREE.MeshBasicMaterial({
          color: 0x66e0ff, wireframe: true, transparent: true, opacity: 0.28 }));
        w.userData.holoWire = true; w.raycast = () => {};   // never intercept point-to-inspect rays
        o.add(w); o.userData.wire = w;
      }
      o.material = o.userData.holoMat; o.userData.wire.visible = true;
    } else if (o.userData.origMat) {
      o.material = o.userData.origMat;
      if (o.userData.wire) o.userData.wire.visible = false;
    }
  });
  applyClip(root);   // the skin swaps in fresh materials, which arrive without the section plane
}

// ---------- hand skeleton drawn INTO the 3D scene (your hand appears inside the hologram) ----------
const MAX_JOINTS = 42;               // 2 hands * 21 landmarks
const MAX_BONES = 2 * 24;            // 2 hands * (HAND_CONNECTIONS is 21, 24 is safe headroom)
const joints = new THREE.InstancedMesh(
  new THREE.SphereGeometry(0.045, 12, 12),
  new THREE.MeshBasicMaterial({ color: 0xffffff }), MAX_JOINTS);   // per-instance color set in updateHandViz
joints.frustumCulled = false; scene.add(joints);
const JOINT_LIVE = new THREE.Color(0x7CFFB2), JOINT_PARK = new THREE.Color(0xff5a5a); // green = live, red = parked (fist)
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
    const col = handPose(hand) === 'fist' ? JOINT_PARK : JOINT_LIVE;     // parked hand's joints glow red
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
  pivot.remove(model); disposeTree(model);
  model = obj; fitToView(model); pivot.add(model);
  collectParts(model); applyHoloSkin(model, holoSkin);
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
  const entry = LOADERS[ext];
  if (!entry) { errEl.textContent = `Unsupported file “.${ext}” — use .glb, .gltf, .obj, .fbx or .stl`; e.target.value = ''; return; }
  errEl.textContent = ''; stateEl.textContent = `Loading ${f.name}…`;
  const url = URL.createObjectURL(f);
  new entry.L().load(url, (res) => {
    swapModel(entry.pick(res)); URL.revokeObjectURL(url);
    stateEl.textContent = `Loaded ${f.name}`; blip(720, 0.14, 'triangle');
    e.target.value = '';   // clear it, or re-picking the SAME file fires no change event and looks broken
  }, undefined, (err) => { URL.revokeObjectURL(url); e.target.value = ''; errEl.textContent = 'Could not load model: ' + err; });
});

// ---------- HUD controls: holo skin · showcase toggles · snapshot · live calibration ----------
$('tHolo').addEventListener('change', (e) => { holoSkin = e.target.checked; applyHoloSkin(model, holoSkin); });
$('tSpin').addEventListener('change', (e) => { autoSpin = e.target.checked; });
$('tBloom').addEventListener('change', (e) => { bloom.enabled = e.target.checked; });
$('tGrid').addEventListener('change', (e) => { grid.visible = reticle.visible = e.target.checked; });
$('snap').addEventListener('click', () => {                 // one-tap PNG of the holo render (preserveDrawingBuffer)
  initAudio(); blip(900, 0.08, 'triangle');
  const a = document.createElement('a');
  a.download = 'holocontrol-' + Date.now() + '.png';
  a.href = renderer.domElement.toDataURL('image/png'); a.click();
});
$('secBtn').addEventListener('click', () => setSection(!sectionOn));
$('tuneBtn').addEventListener('click', () => { initAudio(); $('tune').classList.toggle('hidden'); });
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
};
addEventListener('keydown', (e) => {
  // Let the browser have its own chords, and don't steal keys from a focused control (the sliders
  // are arrow-key operated, and Space/Enter on a focused button must still press it).
  if (e.metaKey || e.ctrlKey || e.altKey || e.target?.closest?.('input, select, textarea, button')) return;
  const fn = KEYS[e.key.length === 1 ? e.key.toLowerCase() : e.key];
  if (!fn) return;
  e.preventDefault(); fn(e.shiftKey ? 0.3 : 0.1); manual();
});

// ---------- point-to-inspect: raycast from the camera through the index fingertip ----------
// The fingertip is drawn into the scene at a fixed plane; a ray from the camera through it
// continues on to whatever part sits behind it — so aiming reads exactly as it looks on screen.
const labelEl = $('label');
const raycaster = new THREE.Raycaster();
const _tip = new THREE.Vector3(), _proj = new THREE.Vector3();
const _tipW = new THREE.Vector3(), _d = new THREE.Vector3(), _binW = new THREE.Vector3();  // grabbed-part fingertip tracking (pivot-local) + bin proximity (world)
let highlighted = null;
function findPart(obj) { while (obj) { if (partSet.has(obj)) return obj; obj = obj.parent; } return null; }
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
function pointAt(hand, stickyId = null) {         // highlight+label the aimed part (blue); returns the part or null
  const w = landmarkToWorld(hand[INDEX_TIP], HAND_SPAN, HAND_DEPTH);
  raycaster.set(camera.position, _tip.set(w.x, w.y, w.z).sub(camera.position).normalize());
  // Walk the hits rather than taking the first: three.js raycasts invisible geometry happily, so
  // without this the ray would keep "hitting" parts you already removed. And prefer the sticky target
  // (the part we're already dwelling on) if the ray still touches it at all, so a 5s hold survives the
  // ray jittering onto a neighbour for a frame — otherwise the dwell would keep resetting near the end.
  let part = null, hit = null;
  for (const h of raycaster.intersectObject(pivot, true)) {
    const p = findPart(h.object);
    if (!p || !p.visible) continue;
    if (!part) { part = p; hit = h; }
    if (stickyId != null && p.userData.pid === stickyId) { part = p; hit = h; break; }
  }
  setHighlight(part);
  if (!part) { labelEl.style.opacity = '0'; return null; }
  _proj.copy(hit.point).project(camera);
  const x = (_proj.x * 0.5 + 0.5) * innerWidth, y = (-_proj.y * 0.5 + 0.5) * innerHeight;
  labelEl.innerHTML = `${part.userData.label}<span class="lsub">${part.userData.info}</span>`;   // name + honest info card, floated at the part
  labelEl.style.transform = `translate(-50%,-150%) translate(${x}px,${y}px)`;
  labelEl.style.opacity = '1';
  return part;
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
let aim = AIM_OFF, locked = null, grabTip = null;   // part-removal machine: state, the grabbed part, and the fingertip pos (pivot-local) where the grab began
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
  // A fist "parks" a hand — we ignore it — so you can drive one-hand gestures with your other
  // hand while both stay comfortably in frame (no yanking a hand out of view to change modes).
  // Two fists = pause everything. We classify each hand once, then act on the live ones only.
  const classified = hands.map(h => ({ h, pose: handPose(h) }));
  const active = classified.filter(a => a.pose !== 'fist');
  const parked = hands.length - active.length;
  dragging = false;                                                        // set true only while rotating
  poseEl.textContent = classified.map((a, i) => `H${i + 1} ${a.pose} ${pinchStrength(a.h).toFixed(2)}`).join('    ');

  const solo = active.length === 1 ? active[0] : null;
  const soloC = solo ? handCenter(solo.h) : null;

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

  // ---- Part removal, pinch-free, stepped every frame BEFORE anything else can claim the hand. Point at
  // a part and hold to grab it onto your fingertip (a long dwell, so a stray frame can't); then just move
  // your hand — the part rides the fingertip 1:1, and a fast flick or pulling it clear and letting go
  // dissolves it. No pinch is involved, which is the whole fix: bringing the thumb in to pinch used to be
  // misread as the grab mid-motion. Stepping unconditionally is what makes a release always land, even if
  // the hand leaves frame mid-pull. Only raycast while the machine is still choosing a target.
  const tip = solo ? landmarkToWorld(solo.h[INDEX_TIP], HAND_SPAN, HAND_DEPTH) : null;
  if (tip) { _tipW.set(tip.x, tip.y, tip.z); pivot.worldToLocal(_tipW); }     // fingertip in the model's own space
  const stickyId = aim.phase === 'aim' ? aim.id : null;                       // keep the dwell locked to the part we started on
  const aimedPart = solo && solo.pose === 'point' && aim.phase !== 'grab' ? pointAt(solo.h, stickyId) : null;
  let inBin = false;
  if (aim.phase === 'grab' && locked && tip && grabTip) {                     // grabbed: the part rides the fingertip 1:1
    _d.copy(_tipW).sub(grabTip);                                             // how far the finger has travelled since the grab
    if (_d.lengthSq() > 1e-8) { locked.userData.pdir.copy(_d).normalize(); locked.userData.pullT = _d.length(); }  // move left => part goes left
    // is the part now inside the dustbin? test in WORLD space (the bin doesn't rotate with the model)
    _binW.copy(locked.userData.home).addScaledVector(locked.userData.dir, current.explode * EXPLODE_K).addScaledVector(locked.userData.pdir, locked.userData.pull);
    inBin = pivot.localToWorld(_binW).distanceTo(bin.position) < BIN_RADIUS;
  }
  aim = aimStep(aim, { pose: solo?.pose ?? null, id: aimedPart?.userData.pid ?? null, now: performance.now(),
                       present: !!solo, inBin }, { dwellMs: GRAB_DWELL_MS });
  if (aim.action === 'grab') { locked = parts[aim.id]; setHighlight(locked, HL_GRAB); ghostOthers(locked); grabTip = _tipW.clone(); showBin(true); }
  else if (aim.action) {                                                     // 'remove' (binned) or 'drop' (let go)
    if (aim.action === 'remove') sinkPart(locked);
    else if (locked) locked.userData.pullT = 0;                            // dropped: eases back home
    locked = null; grabTip = null; ghostOthers(null); showBin(false); clearHighlight();
  }

  if (active.length >= 2) {                                                  // two live hands
    const spread = twoHandSpread(active[0].h, active[1].h);
    const ang = twoHandAngle(active[0].h, active[1].h);                      // twist both hands (like a wheel) => roll
    const zooming = active[0].pose === 'pinch' && active[1].pose === 'pinch';
    // Roll ONLY when not zooming: a two-hand pinch is a pure scale, so the model won't drift in
    // orientation while you resize it (and axis locks then actually hold during a zoom). The
    // deadzone drops sub-threshold angle wobble so only a deliberate twist rolls.
    target.rz += rollDelta(prevAngle, ang, { zooming, deadzone: TWIST_DEADZONE, mirror: MIRROR_X }); // 0 while zooming/first frame
    prevAngle = ang;                                                         // tracked even while zooming, so no jump on release
    prevCenter = null; idle = false; clearHighlight();
    if (zooming) {                                                           // two pinches => pure zoom
      if (prevSpread != null) target.scale = clamp(target.scale + (spread - prevSpread) * ZOOM_SPEED, SCALE_MIN, SCALE_MAX);
      prevSpread = spread;
      stateEl.textContent = 'Zoom'; setMode('zoom');
    } else {                                                                 // else => explode by gap (+ twist-roll)
      target.explode = clamp((spread - EXPLODE_MIN) / (EXPLODE_MAX - EXPLODE_MIN), 0, 1);
      prevSpread = null;
      stateEl.textContent = target.explode > 0.05 ? `Exploded ${Math.round(target.explode * 100)}%` : 'Spread to explode · twist to roll';
      setMode('explode');
    }
  } else if (aim.phase === 'grab') {                                         // part is riding the fingertip (pullT set above)
    prevCenter = null; prevSpread = null; prevAngle = null; idle = false;
    setHighlight(locked, HL_GRAB); labelEl.style.opacity = '0';             // info card was for inspecting; the HUD line drives the grab
    stateEl.textContent = inBin
      ? `🗑 drop it — ${locked?.userData.label} over the bin`
      : `✊ ${locked?.userData.label} follows your finger — carry it to the 🗑 to remove`;
    setMode('grab');
  } else if (aim.phase === 'aim') {                                          // inspecting: info card showing, dwell filling
    prevCenter = null; prevSpread = null; prevAngle = null; idle = false;
    stateEl.textContent = `ℹ ${aimedPart?.userData.label} · ${aimedPart?.userData.info}  ${bar(aim.progress)} hold to grab`;
    setMode('inspect');
  } else if (solo && solo.pose === 'pinch') {                                // one pinch => rotate
    const c = soloC;
    if (prevCenter) {
      const dry = (c.x - prevCenter.x) * ROT_SPEED * MIRROR_X;
      const drx = (c.y - prevCenter.y) * ROT_SPEED;
      target.ry += dry; target.rx += drx;
      spinVel.ry = spinVel.ry * (1 - SPIN_CAPTURE) + dry * SPIN_CAPTURE;    // recent-weighted flick speed
      spinVel.rx = spinVel.rx * (1 - SPIN_CAPTURE) + drx * SPIN_CAPTURE;
    }
    prevCenter = c; prevSpread = null; prevAngle = null; idle = false; dragging = true; clearHighlight();
    stateEl.textContent = `Rotate (pinch ${pinchStrength(solo.h).toFixed(2)})`; setMode('rotate');
  } else if (solo && solo.pose === 'point') {                                // pointing at empty space
    prevCenter = null; prevSpread = null; prevAngle = null; idle = false; clearHighlight();
    stateEl.textContent = 'Point at a part'; setMode('inspect');
  } else {                                                                   // nothing live => drift
    prevCenter = null; prevSpread = null; prevAngle = null; idle = true; clearHighlight();
    stateEl.textContent = parked >= 2 ? '✊ paused'
      : parked ? '✊ parked — other hand is free'
      : hands.length ? 'Pinch = rotate · point = inspect · ✊ = park'
      : 'Show a hand';
    setMode(parked >= 2 ? 'pause' : parked ? 'park' : 'idle');
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
    drawHands(hands);
    applyGestures(hands);
    updateHandViz(hands);
  }
  if (!dragging && (spinVel.rx || spinVel.ry)) {   // flick momentum: keep spinning, then ease to rest
    target.ry += spinVel.ry; target.rx += spinVel.rx;
    spinVel.ry *= SPIN_FRICTION; spinVel.rx *= SPIN_FRICTION;
    if (Math.hypot(spinVel.rx, spinVel.ry) < SPIN_MIN) spinVel.rx = spinVel.ry = 0;
  } else if (idle && autoSpin && performance.now() > manualT) target.ry += IDLE_SPIN;   // gentle turntable so the hologram feels alive
  if (lockX) target.rx = current.rx;   // held axes stop accumulating, so unlocking resumes smoothly (no snap)
  if (lockY) target.ry = current.ry;
  if (lockZ) target.rz = current.rz;
  current.rx += (target.rx - current.rx) * SMOOTH; // ease toward target every frame
  current.ry += (target.ry - current.ry) * SMOOTH;
  current.rz += (target.rz - current.rz) * SMOOTH;
  current.scale += (target.scale - current.scale) * SMOOTH;
  current.explode += (target.explode - current.explode) * SMOOTH;
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
    wireMat.opacity = skin.opacity * 0.55; bin.visible = skin.opacity > 0.02; bin.rotation.y += 0.012;
  }
  composer.render();
}
updateStats(0);   // seed the HUD readout (rocket = 7 parts) before the first frame
loop();
