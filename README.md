# HoloControl

Reach into your webcam and **grab a 3D model out of the air** — pinch to spin it, point to inspect a part,
pull two hands apart to blow it into an exploded diagram. No mouse, no touchpad, no LLM. Just a laptop camera
and hand-written geometry.

Think of the holographic scenes where someone turns a model over in mid-air with their hands. This is a small,
real, running version of that, and every gesture is decided by math I can explain line by line.

<!-- Add a screenshot or GIF here for the submission: save one to docs/screenshot.png (a hand mid-gesture
     reads best — grab a frame from the demo video) and swap in ![HoloControl](docs/screenshot.png) -->


## Why this isn't an "AI wrapper"

There is no chatbot and no API key here. A pre-trained hand-tracking model (MediaPipe HandLandmarker) is used
only as a **sensor** — it hands me 21 points per hand and nothing else. Everything that turns those points into
*control* is geometry I wrote by hand in [`gestures.js`](gestures.js) and [`main.js`](main.js): what counts as a
pinch, how a moving hand becomes rotation, how the gap between two hands becomes zoom, how a pointing finger
picks a part out of the model, and how the jitter gets smoothed so it feels intentional. **That math is the
project** — the model can't do any of it.

## Gestures

| Gesture | How | What it does |
|---|---|---|
| **Rotate** | one hand, thumb + index pinched, move it | spins the model; let go with a flick and it keeps spinning, then eases to rest (momentum) |
| **Inspect** | one hand pointing (index only) | casts a ray from your fingertip into the model and highlights + labels the part you point at |
| **Zoom** | *both* hands pinched | the change in the gap between them scales the model |
| **Explode / rebuild** | both hands up, spread apart / together | blows the model into its component parts and reassembles it — a live exploded diagram |
| **Park** ✊ | make a fist | that hand is ignored, so you can rest it in frame while the other hand drives |
| **Pause** ✊✊ | both fists | freezes everything |

Parking is the trick that makes two hands comfortable: to switch from a two-hand gesture back to one-hand
rotate, you don't yank a hand off-screen — you just close it into a fist and it drops out of the math.

## How it works

```
webcam ─▶ MediaPipe HandLandmarker ─▶ pose + gesture math ─▶ three.js transform ─▶ screen
         (21 landmarks per hand)      gestures.js / main.js   rotate · scale · explode · raycast
```

The parts that were actually hard, and how they're solved:

- **Pinch that works at any distance.** Thumb-tip↔index-tip distance is divided by the palm size, so a pinch
  reads the same whether your hand is near the camera or far — otherwise it only triggers at one depth.
- **A webcam is a mirror.** Moving your hand right has to turn the model right, so X is flipped (`MIRROR_X`) in
  both the rotation and the on-screen hand skeleton, or the whole thing feels backwards.
- **One camera can't measure depth.** So "push your hand toward the screen to zoom" isn't reliable — the gap
  between two hands is used instead. A hardware limit that shaped the interaction design.
- **Raw landmarks jitter.** Gestures set a *target* transform; every frame the *current* transform eases toward
  it (a low-pass filter). This is the whole difference between janky and deliberate.
- **Pose classification.** Each hand is sorted into `fist / pinch / point / open` from which fingers are
  extended, which gives discrete modes (park, pause, inspect) with no buttons on screen.

## The hologram layer

The look is real three.js, not a filter: `UnrealBloom` post-processing for the glow, additive-blended
[fingertip trails](main.js), an `InstancedMesh` hand skeleton recolored per hand (green live, red when parked),
a rotating projector reticle, a CSS scanline + vignette overlay, a boot intro card, and Web Audio synth blips
generated on the fly for each gesture change (no sound files). A live pose HUD shows what each hand is doing.

## Run it locally

A webcam needs a secure context, which `localhost` provides:

```bash
python3 -m http.server 8000
```

Open <http://localhost:8000>, click **Start camera**, and allow access. Drop in your own model with
**Load 3D model** — `.glb`, `.gltf`, `.obj`, `.fbx`, or `.stl`. It's read locally; nothing is uploaded.
(Blender file? Export ▸ glTF 2.0 (.glb).) With no model loaded you get a built-in rocket to play with.

## Test the gesture math

The geometry in `gestures.js` is pure — no camera, no DOM — so it runs and checks itself in Node:

```bash
node gestures.test.js
```

## Tuning

Real cameras and hands vary, so [`main.js`](main.js) opens with a block of constants meant to be tweaked:
`ROT_SPEED`, `ZOOM_SPEED`, `SMOOTH` (lower = smoother but laggier), `MIRROR_X` (flip if rotation feels
backwards), `HAND_SPAN`/`HAND_DEPTH` (how the hand maps into the scene), and `EXPLODE_K`. The physical world
needs a calibration knob a clean model can't guess.

## Stack

Vanilla JS + [three.js](https://threejs.org) + [MediaPipe Tasks Vision](https://ai.google.dev/edge/mediapipe),
all from a CDN via an import map. No build step, no server, no dependencies to install.

## What I learned

I came in knowing basic JS and wanting to understand three.js and how gesture control actually works under the
hood. The things that taught me the most: making a pinch scale-invariant by normalizing to palm size; realizing
a single camera gives you no trustworthy depth and redesigning zoom around that; that smoothing (a target you
ease toward, not the raw value) is what makes an interface feel real; capturing recent velocity to get
flick-to-spin momentum; raycasting a fingertip into a mesh to name the part under it; and using an
`InstancedMesh` so redrawing 42 hand joints every frame stays cheap. Most of the debugging was physical — hands
and cameras don't behave like the tidy diagram, so almost every constant above exists because something felt
wrong until I tuned it.

## AI-use disclosure

Per the hackathon rules: I used an AI coding assistant (Claude) to scaffold the three.js/MediaPipe boilerplate
and to talk through the gesture-mapping design. I reviewed, ran, and tuned all of it, and I can explain every
part — the pinch metric, the two-hand zoom, the pose classifier, and the smoothing were the specific things I
set out to understand. The commit history shows how it was built up piece by piece.
