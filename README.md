# Meta VR & Horizon OS Experiments (`meta-vr`)

A collection of VR / Mixed Reality experiments targeting **Meta VR Glasses** and **Meta Horizon OS**.

## Projects

### 1. [`spatial-web-app`](./spatial-web-app) — House of Cards VR (Immersive Web SDK)
A WebXR spatial hologram visualization of Radiohead's *"House of Cards"* LiDAR point cloud dataset, built with **Meta's Immersive Web SDK (`@iwsdk/core`)**, **Three.js**, and **Horizon UIKitML**.

* **Meta VR Glasses Ready:** Configured for **Gaze + Pinch** (`gazeTracking`, `handTracking`, and `useHandPinchForGrab`) plus the `metaVRGlasses` emulator profile.
* **Single Draw-Call Custom Shader:** Renders up to 15,000 LiDAR points per frame at 30 FPS synced to audio using `THREE.BufferGeometry` (`DynamicDrawUsage`) and a depth-graded neon `THREE.ShaderMaterial`.
* **Interactive 3D Hologram:** Look at and pinch the singing 3D LiDAR bust to grab and reposition it in 3D space, or use the floating Spatial UI Control Deck to play/pause audio, randomize neon palettes, and reset position.

#### Setup & Running (`spatial-web-app`)
1. Link or copy the quantized LiDAR chunk data (`chunks/` and `HouseOfCards_DataSample.mp3` from [`woakin/house-of-cards-webgl`](https://github.com/woakin/house-of-cards-webgl)) into `spatial-web-app/public/data`.
2. Install and start the IWSDK managed development server:
   ```bash
   cd spatial-web-app
   npm install
   npm run dev
   ```
