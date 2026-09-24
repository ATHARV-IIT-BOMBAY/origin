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
import { handCenter, pinchStrength, twoHandSpread, landmarkToWorld, handPose, INDEX_TIP } from './gestures.js';

// ---- Tuning knobs. A webcam is a messy sensor; these are the calibration dials. ----
const ROT_SPEED = 6.0;    // how far a hand move rotates the model
const ZOOM_SPEED = 6.0;   // how strongly two-hand spread scales the model
const SMOOTH = 0.20;      // 0..1 low-pass ease; lower = smoother but laggier
const MIRROR_X = -1;      // flip so moving your hand right rotates the model right
const SCALE_MIN = 0.3, SCALE_MAX = 4.0;
const HAND_SPAN = 4.2;    // how wide the tracked hand maps into the 3D scene
const HAND_DEPTH = 1.5;   // how strongly landmark depth pushes hand joints in/out
const IDLE_SPIN = 0.0015; // lazy auto-rotate (rad/frame) when you're not controlling it
// Flick-to-spin: while pinch-dragging we track the swipe velocity; on release the model keeps
// spinning and eases to rest (friction). CAPTURE = how much the latest frame feeds the estimate.
const SPIN_FRICTION = 0.96, SPIN_CAPTURE = 0.5, SPIN_MIN = 0.001;
const EXPLODE_K = 1.6;                        // full-explosion expansion: parts push out from center
const EXPLODE_MIN = 0.20, EXPLODE_MAX = 0.75; // two-hand spread range mapped onto 0..1 explosion
const BASE_EMISSIVE = 0.6, HIGHLIGHT_EMISSIVE = 2.4; // part glow: resting vs. aimed-at
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm';

const $ = (id) => document.getElementById(id);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const video = $('video'), overlay = $('overlay'), stateEl = $('state'), errEl = $('err');

// ---------- three.js scene ----------
const renderer = new THREE.WebGLRenderer({ canvas: $('three'), antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
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
const grid = new THREE.GridHelper(20, 40, 0x1e6fff, 0x0a2a4a);
grid.position.y = -1.6; scene.add(grid);

// Holo-projector base: two counter-rotating tech rings under the model (bloom makes them glow).
const reticle = new THREE.Group(); reticle.position.y = -1.55; scene.add(reticle);
const _up = new THREE.Vector3(0, 1, 0);
for (const [rIn, rOut, dir] of [[1.70, 1.86, 1], [1.96, 2.02, -1]]) {
  const ring = new THREE.Mesh(new THREE.RingGeometry(rIn, rOut, 96),
    new THREE.MeshBasicMaterial({ color: 0x2aa0ff, transparent: true, opacity: 0.55, side: THREE.DoubleSide }));
  ring.rotation.x = -Math.PI / 2; ring.userData.dir = dir; reticle.add(ring);
}

const pivot = new THREE.Group(); // we rotate/scale this; the model lives inside it
scene.add(pivot);

// Hologram-styled part: translucent lit core + bright wireframe overlay (the bit that blooms).
// Grouped so a whole part can be raycast, highlighted, and flown out as a single unit.
function holoPart(geo, name, label, color = 0x0a2a4a, emissive = 0x0aa0ff) {
  const core = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
    color, emissive, emissiveIntensity: BASE_EMISSIVE,
    metalness: 0.3, roughness: 0.35, transparent: true, opacity: 0.8 }));
  const wire = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
    color: 0x66e0ff, wireframe: true, transparent: true, opacity: 0.45 }));
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

function fitToView(obj) { // center at origin and normalize size so any model frames nicely
  const box = new THREE.Box3().setFromObject(obj);
  const size = box.getSize(new THREE.Vector3());
  obj.position.sub(box.getCenter(new THREE.Vector3()));
  obj.scale.multiplyScalar(1.8 / (Math.max(size.x, size.y, size.z) || 1));
}
fitToView(model);

// Parts = the model's direct children that carry geometry. Each remembers its resting
// position (home); the explode gesture pushes every part radially out from the model's center
// (an exploded-view expansion) and snaps them home. Parts sitting at dead-center don't travel.
let parts = [], partSet = new Set();
function firstMesh(o) { let m = null; o.traverse(c => { if (!m && c.isMesh) m = c; }); return m; }
function collectParts(root) {
  parts = root.children.filter(firstMesh);
  partSet = new Set(parts);
  for (const p of parts) {
    p.userData.home = p.position.clone();
    if (!p.userData.core) p.userData.core = firstMesh(p);
    if (p.userData.label == null) p.userData.label = p.name || 'Part';
  }
}
collectParts(model);

// ---------- hand skeleton drawn INTO the 3D scene (your hand appears inside the hologram) ----------
const MAX_JOINTS = 42;               // 2 hands * 21 landmarks
const MAX_BONES = 2 * 24;            // 2 hands * (HAND_CONNECTIONS is 21, 24 is safe headroom)
const joints = new THREE.InstancedMesh(
  new THREE.SphereGeometry(0.045, 12, 12),
  new THREE.MeshBasicMaterial({ color: 0x7CFFB2 }), MAX_JOINTS);
joints.frustumCulled = false; scene.add(joints);
const boneGeo = new THREE.BufferGeometry();
boneGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(MAX_BONES * 2 * 3), 3));
const bones = new THREE.LineSegments(boneGeo, new THREE.LineBasicMaterial({ color: 0x7CFFB2, transparent: true, opacity: 0.85 }));
bones.frustumCulled = false; scene.add(bones);
const _m = new THREE.Matrix4(), _hidden = new THREE.Matrix4().makeScale(0, 0, 0);
function hideHands() {
  for (let k = 0; k < MAX_JOINTS; k++) joints.setMatrixAt(k, _hidden);
  joints.instanceMatrix.needsUpdate = true; boneGeo.setDrawRange(0, 0);
}
function updateHandViz(hands) {
  const conns = HandLandmarker.HAND_CONNECTIONS, pos = boneGeo.attributes.position.array;
  let j = 0, v = 0; // joint index, bone-vertex index
  for (const hand of hands) {
    const world = hand.map(p => landmarkToWorld(p, HAND_SPAN, HAND_DEPTH));
    for (const p of world) if (j < MAX_JOINTS) { _m.makeTranslation(p.x, p.y, p.z); joints.setMatrixAt(j++, _m); }
    for (const c of conns) if (v + 2 <= MAX_BONES * 2) {
      const a = world[c.start], d = world[c.end];
      pos.set([a.x, a.y, a.z, d.x, d.y, d.z], v * 3); v += 2;
    }
  }
  for (let k = j; k < MAX_JOINTS; k++) joints.setMatrixAt(k, _hidden); // hide unused instances
  joints.instanceMatrix.needsUpdate = true;
  boneGeo.setDrawRange(0, v); boneGeo.attributes.position.needsUpdate = true;
}
hideHands();

function resize() {
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
function swapModel(obj) {
  pivot.remove(model);
  model = obj; fitToView(model); pivot.add(model);
  collectParts(model); clearHighlight();
  target.explode = current.explode = 0; target.rx = target.ry = 0; spinVel.rx = spinVel.ry = 0;
}
$('file').addEventListener('change', (e) => {
  const f = e.target.files[0]; if (!f) return;
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
    stateEl.textContent = `Loaded ${f.name}`;
  }, undefined, (err) => { URL.revokeObjectURL(url); errEl.textContent = 'Could not load model: ' + err; });
});

// ---------- transform state: gestures set `target`, each frame eases `current` toward it ----------
const target = { rx: 0, ry: 0, scale: 1, explode: 0 };
const current = { rx: 0, ry: 0, scale: 1, explode: 0 };
$('reset').addEventListener('click', () => { target.rx = 0; target.ry = 0; target.scale = 1; target.explode = 0; spinVel.rx = spinVel.ry = 0; });

// ---------- point-to-inspect: raycast from the camera through the index fingertip ----------
// The fingertip is drawn into the scene at a fixed plane; a ray from the camera through it
// continues on to whatever part sits behind it — so aiming reads exactly as it looks on screen.
const labelEl = $('label');
const raycaster = new THREE.Raycaster();
const _tip = new THREE.Vector3(), _proj = new THREE.Vector3();
let highlighted = null;
function findPart(obj) { while (obj) { if (partSet.has(obj)) return obj; obj = obj.parent; } return null; }
function setHighlight(part) {
  if (highlighted && highlighted !== part) {
    const m = highlighted.userData.core?.material;
    if (m && 'emissiveIntensity' in m) m.emissiveIntensity = BASE_EMISSIVE;
  }
  highlighted = part;
  const m = part?.userData.core?.material;
  if (m && 'emissiveIntensity' in m) m.emissiveIntensity = HIGHLIGHT_EMISSIVE;
}
function clearHighlight() { setHighlight(null); labelEl.style.opacity = '0'; }
function pointAt(hand) {                          // highlight+label the aimed part; return its label or null
  const w = landmarkToWorld(hand[INDEX_TIP], HAND_SPAN, HAND_DEPTH);
  raycaster.set(camera.position, _tip.set(w.x, w.y, w.z).sub(camera.position).normalize());
  const hit = raycaster.intersectObject(pivot, true)[0];
  const part = hit ? findPart(hit.object) : null;
  setHighlight(part);
  if (!part) { labelEl.style.opacity = '0'; return null; }
  _proj.copy(hit.point).project(camera);
  const x = (_proj.x * 0.5 + 0.5) * innerWidth, y = (-_proj.y * 0.5 + 0.5) * innerHeight;
  labelEl.textContent = part.userData.label;
  labelEl.style.transform = `translate(-50%,-150%) translate(${x}px,${y}px)`;
  labelEl.style.opacity = '1';
  return part.userData.label;
}

// ---------- MediaPipe hand tracking ----------
let handLandmarker = null, drawUtils = null, running = false, lastVideoTime = -1;
let prevCenter = null, prevSpread = null, idle = true;
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

async function startCamera() {
  try {
    errEl.textContent = '';
    if (!handLandmarker) { stateEl.textContent = 'Loading model…'; await initHands(); }
    video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    await video.play();
    overlay.width = video.videoWidth; overlay.height = video.videoHeight;
    running = true; stateEl.textContent = 'Show a hand';
  } catch (err) { errEl.textContent = 'Camera/model error: ' + err; }
}
$('start').addEventListener('click', startCamera);

function applyGestures(hands) {
  // A fist "parks" a hand — we ignore it — so you can drive one-hand gestures with your other
  // hand while both stay comfortably in frame (no yanking a hand out of view to change modes).
  // Two fists = pause everything. We classify each hand once, then act on the live ones only.
  const active = hands.map(h => ({ h, pose: handPose(h) })).filter(a => a.pose !== 'fist');
  const parked = hands.length - active.length;
  dragging = false;                                                        // set true only while rotating

  if (active.length >= 2) {                                                  // two live hands
    const spread = twoHandSpread(active[0].h, active[1].h);
    prevCenter = null; idle = false; clearHighlight();
    if (active[0].pose === 'pinch' && active[1].pose === 'pinch') {          // two pinches => zoom
      if (prevSpread != null) target.scale = clamp(target.scale + (spread - prevSpread) * ZOOM_SPEED, SCALE_MIN, SCALE_MAX);
      prevSpread = spread;
      stateEl.textContent = 'Zoom';
    } else {                                                                 // else => explode by gap
      target.explode = clamp((spread - EXPLODE_MIN) / (EXPLODE_MAX - EXPLODE_MIN), 0, 1);
      prevSpread = null;
      stateEl.textContent = target.explode > 0.05 ? `Exploded ${Math.round(target.explode * 100)}%` : 'Spread hands to explode';
    }
  } else if (active.length === 1 && active[0].pose === 'pinch') {            // one pinch => rotate
    const c = handCenter(active[0].h);
    if (prevCenter) {
      const dry = (c.x - prevCenter.x) * ROT_SPEED * MIRROR_X;
      const drx = (c.y - prevCenter.y) * ROT_SPEED;
      target.ry += dry; target.rx += drx;
      spinVel.ry = spinVel.ry * (1 - SPIN_CAPTURE) + dry * SPIN_CAPTURE;    // recent-weighted flick speed
      spinVel.rx = spinVel.rx * (1 - SPIN_CAPTURE) + drx * SPIN_CAPTURE;
    }
    prevCenter = c; prevSpread = null; idle = false; dragging = true; clearHighlight();
    stateEl.textContent = `Rotate (pinch ${pinchStrength(active[0].h).toFixed(2)})`;
  } else if (active.length === 1 && active[0].pose === 'point') {            // one point => inspect
    prevCenter = null; prevSpread = null; idle = false;
    const name = pointAt(active[0].h);
    stateEl.textContent = name ? `▶ ${name}` : 'Point at a part';
  } else {                                                                   // nothing live => drift
    prevCenter = null; prevSpread = null; idle = true; clearHighlight();
    stateEl.textContent = parked >= 2 ? '✊ paused'
      : parked ? '✊ parked — other hand is free'
      : hands.length ? 'Pinch = rotate · point = inspect · ✊ = park'
      : 'Show a hand';
  }
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
  } else if (idle) target.ry += IDLE_SPIN;         // gentle drift so the hologram feels alive
  current.rx += (target.rx - current.rx) * SMOOTH; // ease toward target every frame
  current.ry += (target.ry - current.ry) * SMOOTH;
  current.scale += (target.scale - current.scale) * SMOOTH;
  current.explode += (target.explode - current.explode) * SMOOTH;
  for (const p of parts) p.position.copy(p.userData.home).multiplyScalar(1 + current.explode * EXPLODE_K);
  pivot.rotation.set(current.rx, current.ry, 0);
  pivot.scale.setScalar(current.scale);
  for (const r of reticle.children) r.rotateOnWorldAxis(_up, 0.004 * r.userData.dir); // flat spin, projector look
  composer.render();
}
loop();
