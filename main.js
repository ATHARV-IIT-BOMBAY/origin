// HoloControl — rotate & zoom a 3D model with webcam hand gestures.
// Flow: webcam -> MediaPipe HandLandmarker -> gesture math (gestures.js) -> three.js transform.
// No LLM, no backend. The hand model is only a *sensor*; the interaction is hand-written geometry.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { HandLandmarker, FilesetResolver, DrawingUtils }
  from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18';
import { handCenter, isPinching, pinchStrength, twoHandSpread } from './gestures.js';

// ---- Tuning knobs. A webcam is a messy sensor; these are the calibration dials. ----
const ROT_SPEED = 6.0;    // how far a hand move rotates the model
const ZOOM_SPEED = 6.0;   // how strongly two-hand spread scales the model
const SMOOTH = 0.20;      // 0..1 low-pass ease; lower = smoother but laggier
const MIRROR_X = -1;      // flip so moving your hand right rotates the model right
const PINCH_ON = 0.6;     // pinch strength (0..1) needed to start rotating
const SCALE_MIN = 0.3, SCALE_MAX = 4.0;
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

const pivot = new THREE.Group(); // we rotate/scale this; the model lives inside it
scene.add(pivot);

function makeDefaultModel() {
  const geo = new THREE.TorusKnotGeometry(0.7, 0.24, 220, 32);
  const mat = new THREE.MeshStandardMaterial({ color: 0x5b8cff, metalness: 0.9, roughness: 0.18 });
  return new THREE.Mesh(geo, mat);
}
let model = makeDefaultModel();
pivot.add(model);

function fitToView(obj) { // center at origin and normalize size so any model frames nicely
  const box = new THREE.Box3().setFromObject(obj);
  const size = box.getSize(new THREE.Vector3());
  obj.position.sub(box.getCenter(new THREE.Vector3()));
  obj.scale.multiplyScalar(1.8 / (Math.max(size.x, size.y, size.z) || 1));
}
fitToView(model);

function resize() {
  renderer.setSize(innerWidth, innerHeight, false);
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
}
addEventListener('resize', resize); resize();

// ---------- load your own model (object URL avoids any CORS/backend) ----------
const loader = new GLTFLoader();
$('file').addEventListener('change', (e) => {
  const f = e.target.files[0]; if (!f) return;
  const url = URL.createObjectURL(f);
  loader.load(url, (gltf) => {
    pivot.remove(model);
    model = gltf.scene; fitToView(model); pivot.add(model);
    URL.revokeObjectURL(url);
  }, undefined, (err) => { errEl.textContent = 'Could not load model: ' + err; });
});

// ---------- transform state: gestures set `target`, each frame eases `current` toward it ----------
const target = { rx: 0, ry: 0, scale: 1 };
const current = { rx: 0, ry: 0, scale: 1 };
$('reset').addEventListener('click', () => { target.rx = 0; target.ry = 0; target.scale = 1; });

// ---------- MediaPipe hand tracking ----------
let handLandmarker = null, drawUtils = null, running = false, lastVideoTime = -1;
let prevCenter = null, prevSpread = null;

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
  if (hands.length >= 2) {                       // two hands => zoom by their separation
    const spread = twoHandSpread(hands[0], hands[1]);
    if (prevSpread != null) target.scale = clamp(target.scale + (spread - prevSpread) * ZOOM_SPEED, SCALE_MIN, SCALE_MAX);
    prevSpread = spread; prevCenter = null;
    stateEl.textContent = 'Zooming';
  } else if (hands.length === 1 && isPinching(hands[0], PINCH_ON)) { // one pinch => rotate by its motion
    const c = handCenter(hands[0]);
    if (prevCenter) {
      target.ry += (c.x - prevCenter.x) * ROT_SPEED * MIRROR_X;
      target.rx += (c.y - prevCenter.y) * ROT_SPEED;
    }
    prevCenter = c; prevSpread = null;
    stateEl.textContent = `Rotating (pinch ${pinchStrength(hands[0]).toFixed(2)})`;
  } else {                                        // idle => release the anchors
    prevCenter = null; prevSpread = null;
    stateEl.textContent = hands.length ? 'Pinch to rotate' : 'Show a hand';
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
  }
  current.rx += (target.rx - current.rx) * SMOOTH; // ease toward target every frame
  current.ry += (target.ry - current.ry) * SMOOTH;
  current.scale += (target.scale - current.scale) * SMOOTH;
  pivot.rotation.set(current.rx, current.ry, 0);
  pivot.scale.setScalar(current.scale);
  renderer.render(scene, camera);
}
loop();
