# HoloControl

Rotate and zoom a 3D model in the browser using **webcam hand gestures** — no mouse, no touchpad, no LLM.
Pinch one hand and move it to spin the model; hold up two hands and move them apart or together to zoom.

Think of the hologram scenes where someone grabs a 3D object out of the air and turns it over — this is a
small, real, running version of that, built with a plain laptop webcam.

## Why this isn't an "AI wrapper"

There is no chatbot and no API key here. A pre-trained hand-tracking model (MediaPipe HandLandmarker) is used
only as a **sensor** — it tells me where 21 points of each hand are. Everything that makes the interaction feel
like control — deciding what a pinch means, converting hand motion into rotation, mapping the gap between two
hands to zoom, and smoothing out the jitter so it doesn't feel drunk — is geometry I wrote by hand in
[`gestures.js`](gestures.js) and [`main.js`](main.js). That math is the project.

## How it works

```
webcam ─▶ MediaPipe HandLandmarker ─▶ gesture math (gestures.js) ─▶ three.js transform ─▶ screen
         (21 landmarks per hand)      pinch / center / spread        rotate & scale a pivot
```

- **Rotate** — one hand, thumb + index pinched together. As the pinched hand moves, the model turns with it.
  Pinch strength is the thumb-tip↔index-tip distance divided by the palm size, so it means the same thing
  whether your hand is close to the camera or far away.
- **Zoom** — two hands on screen. The distance between the two palm centers drives the model's scale; move your
  hands apart to zoom in, bring them together to zoom out. (A single webcam can't measure depth reliably, so
  spreading two hands is used instead of pushing one hand toward the camera.)
- **Smoothing** — raw landmarks are jittery, so gestures set a *target* transform and each frame the *current*
  transform eases toward it (a simple low-pass filter). This is the difference between a demo that looks janky
  and one that looks intentional.

## Run it locally

A webcam needs a secure context, which `localhost` counts as:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000>, click **Start camera**, and allow camera access. You can also drop in your
own `.glb`/`.gltf` file with **Load .glb** — it's read locally, nothing is uploaded.

## Test the gesture math

The geometry in `gestures.js` is pure (no camera, no DOM), so it runs and checks itself in Node:

```bash
node gestures.test.js
```

## Tuning

`main.js` opens with a block of constants (`ROT_SPEED`, `ZOOM_SPEED`, `SMOOTH`, `MIRROR_X`, `PINCH_ON`).
Real cameras and hands vary, so these are meant to be tweaked — if rotation feels backwards, flip `MIRROR_X`;
if it feels twitchy, lower `SMOOTH`.

## Stack

Vanilla JS + [three.js](https://threejs.org) + [MediaPipe Tasks Vision](https://ai.google.dev/edge/mediapipe),
all loaded from a CDN via an import map. No build step, no server, no dependencies to install.

## AI-use disclosure

Per the hackathon rules: I used an AI coding assistant (Claude) to help scaffold the project structure and the
three.js/MediaPipe boilerplate, and to talk through the gesture-mapping design. I reviewed, ran, and tuned all
of it, and I can explain every part — the pinch metric, the two-hand zoom, and the smoothing were the core
things I set out to understand. The commit history reflects how it was built up.
