import {
  createSystem,
  DistanceGrabbable,
  Hovered,
  MovementMode,
  RayInteractable,
  UIKit,
  UIKitMLAsset,
  VisibilityState,
} from '@iwsdk/core';
import * as THREE from 'three';

const FPS = 30;
const MAX_POINTS = 15000;
const INITIAL_PREVIEW_FRAME = 41;
const HOLOGRAM_SCALE = 0.0038;
const DEFAULT_POS = new THREE.Vector3(0, 1.42, -1.48);

interface FrameData {
  numPoints: number;
  byteOffset: number;
  floatOffset?: number;
  frameMinX?: number;
  rangeX?: number;
  frameMinY?: number;
  rangeY?: number;
  frameMinZ?: number;
  rangeZ?: number;
}

interface ParserState {
  buffer: ArrayBuffer | null;
  dataview: DataView | null;
  floatData: Float32Array | null;
  u8View: Uint8Array | null;
  u16View: Uint16Array | null;
  isQuantized: boolean;
  totalFrames: number;
  nextByteOffset: number;
  lastParsedBytes: number;
  frames: FrameData[];
}

function advanceFrameParser(
  state: ParserState,
  effectiveLoadedBytes: number,
): void {
  if (!state.dataview || effectiveLoadedBytes <= state.lastParsedBytes) {
    return;
  }

  const dataview = state.dataview;
  const frames = state.frames;
  const totalFrames = state.totalFrames;
  let byteOffset = state.nextByteOffset;

  if (state.isQuantized) {
    while (frames.length < totalFrames) {
      if (byteOffset + 28 > effectiveLoadedBytes) break;

      const numPoints = dataview.getUint32(byteOffset, true);
      if (numPoints === 0 || numPoints > MAX_POINTS * 2) break;

      const frameSize = 28 + numPoints * 8;
      if (byteOffset + frameSize > effectiveLoadedBytes) break;

      const hOffset = byteOffset + 4;
      const frameMinX = dataview.getFloat32(hOffset, true);
      const frameMaxX = dataview.getFloat32(hOffset + 4, true);
      const frameMinY = dataview.getFloat32(hOffset + 8, true);
      const frameMaxY = dataview.getFloat32(hOffset + 12, true);
      const frameMinZ = dataview.getFloat32(hOffset + 16, true);
      const frameMaxZ = dataview.getFloat32(hOffset + 20, true);

      frames.push({
        numPoints,
        byteOffset: byteOffset + 28,
        frameMinX,
        rangeX: frameMaxX - frameMinX || 1,
        frameMinY,
        rangeY: frameMaxY - frameMinY || 1,
        frameMinZ,
        rangeZ: frameMaxZ - frameMinZ || 1,
      });
      byteOffset += frameSize;
    }
  } else {
    while (frames.length < totalFrames) {
      if (byteOffset + 4 > effectiveLoadedBytes) break;
      const numPoints = dataview.getUint32(byteOffset, true);
      if (numPoints === 0 || numPoints > MAX_POINTS * 2) break;

      const frameSize = 4 + numPoints * 16;
      if (byteOffset + frameSize > effectiveLoadedBytes) break;

      frames.push({
        numPoints,
        byteOffset: byteOffset + 4,
        floatOffset: (byteOffset + 4) / 4,
      });
      byteOffset += frameSize;
    }
  }

  state.nextByteOffset = byteOffset;
  state.lastParsedBytes = effectiveLoadedBytes;
}

function generateNeonPalette(): {
  colorA: string;
  colorB: string;
  colorC: string;
} {
  const hueA = Math.floor(Math.random() * 360);
  const hueB = (hueA + 120 + Math.random() * 60) % 360;
  const hueC = (hueB + 120 + Math.random() * 60) % 360;

  const l = 0.55;
  const s = 1.0;
  const h2k = (h: number) => {
    const f = (n: number) => {
      const k = (n + h / 30) % 12;
      const color =
        l - s * Math.min(l, 1 - l) * Math.max(Math.min(k - 3, 9 - k, 1), -1);
      return Math.round(255 * color)
        .toString(16)
        .padStart(2, '0');
    };
    return `#${f(0)}${f(8)}${f(4)}`;
  };

  return { colorA: h2k(hueA), colorB: h2k(hueB), colorC: h2k(hueC) };
}

export class HouseOfCardsSystem extends createSystem({}) {
  private audio!: HTMLAudioElement;
  private isPlaying = false;
  private loadedBytes = 0;
  private currentFrame = -1;
  private previewTime = 0;

  private parserState: ParserState = {
    buffer: null,
    dataview: null,
    floatData: null,
    u8View: null,
    u16View: null,
    isQuantized: false,
    totalFrames: 0,
    nextByteOffset: 0,
    lastParsedBytes: 0,
    frames: [],
  };

  private geometry!: THREE.BufferGeometry;
  private shaderMaterial!: THREE.ShaderMaterial;
  private hologramGroup!: THREE.Group;
  private proxyRing!: THREE.Mesh;
  private proxyRingMaterial!: THREE.MeshBasicMaterial;
  private hologramEntityId: number | null = null;

  private statusLabel: UIKit.Text | null = null;
  private playButtonLabel: UIKit.Text | null = null;

  init(): void {
    // 1. Set up audio element
    this.audio = new Audio('/data/HouseOfCards_DataSample.mp3');
    this.audio.loop = true;
    this.audio.preload = 'auto';

    // 2. Build Three.js Point Cloud geometry + custom VR ShaderMaterial
    const initialPositions = new Float32Array(MAX_POINTS * 3);
    const initialIntensities = new Float32Array(MAX_POINTS);

    this.geometry = new THREE.BufferGeometry();
    const posAttr = new THREE.BufferAttribute(initialPositions, 3);
    posAttr.setUsage(THREE.DynamicDrawUsage);
    const intAttr = new THREE.BufferAttribute(initialIntensities, 1);
    intAttr.setUsage(THREE.DynamicDrawUsage);

    this.geometry.setAttribute('position', posAttr);
    this.geometry.setAttribute('intensity', intAttr);

    this.shaderMaterial = new THREE.ShaderMaterial({
      uniforms: {
        colorA: { value: new THREE.Color('#4facfe') },
        colorB: { value: new THREE.Color('#00f2fe') },
        colorC: { value: new THREE.Color('#f093fb') },
        pointSize: { value: 3.4 },
      },
      vertexShader: `
        attribute float intensity;
        varying float vIntensity;
        varying float vZ;
        uniform float pointSize;

        void main() {
          vIntensity = intensity;
          vZ = position.z;
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = clamp(pointSize * (2.6 / max(0.25, -mvPosition.z)), 1.5, 11.0);
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        uniform vec3 colorA;
        uniform vec3 colorB;
        uniform vec3 colorC;

        varying float vIntensity;
        varying float vZ;

        void main() {
          if (vZ < -380.0) discard;
          float normalizedInt = clamp(vIntensity / 40.0, 0.0, 1.0);

          vec2 xy = gl_PointCoord.xy - vec2(0.5);
          float ll = length(xy);
          if (ll > 0.5) discard;

          float t = clamp((vZ + 250.0) / 200.0, 0.0, 1.0);

          vec3 mixColor;
          if (t < 0.5) {
            mixColor = mix(colorA, colorB, t * 2.0);
          } else {
            mixColor = mix(colorB, colorC, (t - 0.5) * 2.0);
          }

          float alpha = smoothstep(0.5, 0.0, ll);
          gl_FragColor = vec4(mixColor * (normalizedInt * 0.85 + 0.45), alpha);
        }
      `,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    const points = new THREE.Points(this.geometry, this.shaderMaterial);
    points.frustumCulled = false;
    points.rotation.set(0, 0, Math.PI);
    points.scale.setScalar(HOLOGRAM_SCALE);
    // Center the raw LiDAR coordinates inside the grabbable hologram container
    points.position.set(
      75 * HOLOGRAM_SCALE,
      85 * HOLOGRAM_SCALE,
      145 * HOLOGRAM_SCALE,
    );

    // 3. Create a grabbable hologram container with a subtle base ring & hit sphere
    this.hologramGroup = new THREE.Group();
    this.hologramGroup.name = 'HouseOfCardsHologram';

    // Invisible hit volume so Gaze + Pinch raycasting can easily target the point cloud
    const hitGeo = new THREE.SphereGeometry(0.42, 16, 16);
    const hitMat = new THREE.MeshBasicMaterial({
      transparent: true,
      opacity: 0.0,
      depthWrite: false,
    });
    const hitMesh = new THREE.Mesh(hitGeo, hitMat);
    this.hologramGroup.add(hitMesh);

    // Subtle glowing holographic emitter ring at the base of the bust
    const ringGeo = new THREE.RingGeometry(0.32, 0.35, 48);
    ringGeo.rotateX(-Math.PI / 2);
    this.proxyRingMaterial = new THREE.MeshBasicMaterial({
      color: 0x00f2fe,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.45,
      blending: THREE.AdditiveBlending,
    });
    this.proxyRing = new THREE.Mesh(ringGeo, this.proxyRingMaterial);
    this.proxyRing.position.set(0, -0.42, 0);
    this.hologramGroup.add(this.proxyRing);

    this.hologramGroup.add(points);

    const entity = this.world.createTransformEntity(this.hologramGroup);
    entity.object3D!.position.copy(DEFAULT_POS);
    entity
      .addComponent(RayInteractable)
      .addComponent(DistanceGrabbable, {
        movementMode: MovementMode.MoveFromTarget,
        returnToOrigin: false,
      });
    this.hologramEntityId = entity.index;

    // 4. Hide starter sample clutter so the hologram & control deck take center stage
    for (const id of [
      'robot',
      'robot-contact',
      'select-label',
      'grab-plinth',
      'spin-plinth',
      'gallery-set',
      'sideboard-set',
    ]) {
      const obj = this.world.getSceneObject(id);
      if (obj) {
        obj.visible = false;
      }
    }

    // 5. Wire Spatial UI Panel controls & keyboard shortcuts
    this.wirePanelControls();
    this.wireKeyboardControls();

    // 6. Start streaming the quantized LiDAR chunks
    void this.loadChunkStream();
  }

  private wirePanelControls(): void {
    const panel = this.world.getSceneObject<UIKitMLAsset>('welcome-panel');
    if (!panel) return;

    this.statusLabel = panel.getElementById<UIKit.Text>('status-text') ?? null;
    this.playButtonLabel =
      panel.getElementById<UIKit.Text>('play-label') ?? null;

    const playBtn = panel.getElementById('play-button');
    const vibeBtn = panel.getElementById('vibe-button');
    const resetBtn = panel.getElementById('reset-button');

    const onPlay = () => this.togglePlay();
    const onVibe = () => this.randomizeVibe();
    const onReset = () => this.resetHologram();

    playBtn?.addEventListener('click', onPlay);
    vibeBtn?.addEventListener('click', onVibe);
    resetBtn?.addEventListener('click', onReset);

    this.cleanupFuncs.push(
      () => playBtn?.removeEventListener('click', onPlay),
      () => vibeBtn?.removeEventListener('click', onVibe),
      () => resetBtn?.removeEventListener('click', onReset),
      this.world.visibilityState.subscribe((state) => {
        if (state === VisibilityState.VisibleBlurred && this.isPlaying) {
          this.audio.pause();
          this.isPlaying = false;
          this.updatePlayLabel();
        }
      }),
    );
  }

  private wireKeyboardControls(): void {
    const onKeyDown = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }
      if (e.code === 'KeyP') {
        e.preventDefault();
        this.togglePlay();
      } else if (e.code === 'KeyV') {
        e.preventDefault();
        this.randomizeVibe();
      } else if (e.code === 'KeyR') {
        e.preventDefault();
        this.resetHologram();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    this.cleanupFuncs.push(() =>
      window.removeEventListener('keydown', onKeyDown),
    );
  }

  private updatePlayLabel(): void {
    this.playButtonLabel?.setProperties({
      text: this.isPlaying ? 'Pause Audio & Sync' : 'Play Radiohead Sync',
    });
  }

  private updateStatus(text: string): void {
    this.statusLabel?.setProperties({ text });
  }

  public togglePlay(): void {
    if (this.audio.paused) {
      this.audio
        .play()
        .then(() => {
          this.isPlaying = true;
          this.updatePlayLabel();
        })
        .catch((err) => {
          console.warn('Audio play blocked, advancing procedurally:', err);
          this.isPlaying = !this.isPlaying;
          this.updatePlayLabel();
        });
    } else {
      this.audio.pause();
      this.isPlaying = false;
      this.updatePlayLabel();
    }
  }

  public randomizeVibe(): void {
    const palette = generateNeonPalette();
    this.shaderMaterial.uniforms.colorA.value.set(palette.colorA);
    this.shaderMaterial.uniforms.colorB.value.set(palette.colorB);
    this.shaderMaterial.uniforms.colorC.value.set(palette.colorC);
    this.proxyRingMaterial.color.set(palette.colorB);
  }

  public resetHologram(): void {
    this.hologramGroup.position.copy(DEFAULT_POS);
    this.hologramGroup.rotation.set(0, 0, 0);
  }

  private initParserBuffer(buffer: ArrayBuffer): void {
    const dataview = new DataView(buffer);
    const magicView = new Uint8Array(buffer, 0, 4);
    const isQuant =
      magicView[0] === 0x48 &&
      magicView[1] === 0x4f &&
      magicView[2] === 0x43 &&
      magicView[3] === 0x51; // 'HOCQ'

    const totalFrames = isQuant
      ? dataview.getUint32(4, true)
      : dataview.getUint32(0, true);

    this.parserState = {
      buffer,
      dataview,
      floatData: isQuant ? null : new Float32Array(buffer),
      u8View: new Uint8Array(buffer),
      u16View: new Uint16Array(buffer),
      isQuantized: isQuant,
      totalFrames,
      nextByteOffset: isQuant ? 8 : 4,
      lastParsedBytes: 0,
      frames: [],
    };
  }

  private async loadChunkStream(): Promise<void> {
    try {
      this.updateStatus('Loading LiDAR manifest...');
      const manifestRes = await fetch('/data/chunks/manifest.json');
      if (!manifestRes.ok) {
        throw new Error(`HTTP ${manifestRes.status}`);
      }
      const manifest = (await manifestRes.json()) as {
        totalSize?: number;
        chunks?: { file: string; size?: number }[];
      };

      const totalSize = manifest.totalSize || 204880764;
      const chunkList = manifest.chunks || [];
      if (chunkList.length === 0) {
        throw new Error('Manifest has no chunks');
      }

      const combined = new Uint8Array(totalSize);

      // Load chunk 0 immediately for instant preview
      const chunk0Res = await fetch(chunkList[0].file);
      const chunk0Buf = await chunk0Res.arrayBuffer();
      combined.set(new Uint8Array(chunk0Buf), 0);

      this.initParserBuffer(combined.buffer);
      this.loadedBytes = chunk0Buf.byteLength;
      advanceFrameParser(this.parserState, this.loadedBytes);
      this.renderFrame(
        Math.min(INITIAL_PREVIEW_FRAME, this.parserState.frames.length - 1),
      );
      this.updateStatus(
        `Hologram Ready (${this.parserState.frames.length} frames) - Streaming...`,
      );

      // Stream remaining chunks in the background
      let offset = chunk0Buf.byteLength;
      for (let i = 1; i < chunkList.length; i++) {
        const res = await fetch(chunkList[i].file);
        if (!res.ok) break;
        const buf = await res.arrayBuffer();
        combined.set(new Uint8Array(buf), offset);
        offset += buf.byteLength;
        this.loadedBytes = offset;

        if (i % 4 === 0 || i === chunkList.length - 1) {
          const pct = Math.min(100, Math.round((offset / totalSize) * 100));
          advanceFrameParser(this.parserState, this.loadedBytes);
          this.updateStatus(
            pct === 100
              ? `All ${this.parserState.frames.length} LiDAR Frames Ready`
              : `Streaming LiDAR (${pct}% - ${this.parserState.frames.length} frames)`,
          );
        }
      }
    } catch (err) {
      console.error('Failed to stream House of Cards LiDAR chunks:', err);
      this.updateStatus('LiDAR Load Error - check /public/data');
    }
  }

  private renderFrame(targetFrame: number): void {
    const { frames, isQuantized, u16View, u8View, floatData } =
      this.parserState;
    const frame = frames[targetFrame];
    if (!frame) return;

    const pointsToProcess = Math.min(frame.numPoints, MAX_POINTS);
    const posAttr = this.geometry.getAttribute(
      'position',
    ) as THREE.BufferAttribute;
    const intAttr = this.geometry.getAttribute(
      'intensity',
    ) as THREE.BufferAttribute;

    const positions = posAttr.array as Float32Array;
    const intensities = intAttr.array as Float32Array;

    if (isQuantized && frame.frameMinX !== undefined && u16View && u8View) {
      const frameMinX = frame.frameMinX;
      const frameMinY = frame.frameMinY!;
      const frameMinZ = frame.frameMinZ!;
      const scaleX = frame.rangeX! / 65535;
      const scaleY = frame.rangeY! / 65535;
      const scaleZ = frame.rangeZ! / 65535;

      let pOffset = frame.byteOffset + 6;
      let u16Idx = frame.byteOffset >> 1;
      let outIdx = 0;

      for (let i = 0; i < pointsToProcess; i++) {
        positions[outIdx] = frameMinX + u16View[u16Idx] * scaleX;
        positions[outIdx + 1] = frameMinY + u16View[u16Idx + 1] * scaleY;
        positions[outIdx + 2] = frameMinZ + u16View[u16Idx + 2] * scaleZ;
        intensities[i] = u8View[pOffset];

        u16Idx += 4;
        pOffset += 8;
        outIdx += 3;
      }
    } else if (floatData && frame.floatOffset !== undefined) {
      let floatIdx = frame.floatOffset;
      let outIdx = 0;
      for (let i = 0; i < pointsToProcess; i++) {
        positions[outIdx] = floatData[floatIdx];
        positions[outIdx + 1] = floatData[floatIdx + 1];
        positions[outIdx + 2] = floatData[floatIdx + 2];
        intensities[i] = floatData[floatIdx + 3];

        floatIdx += 4;
        outIdx += 3;
      }
    }

    posAttr.clearUpdateRanges();
    posAttr.addUpdateRange(0, pointsToProcess * 3);
    posAttr.needsUpdate = true;

    intAttr.clearUpdateRanges();
    intAttr.addUpdateRange(0, pointsToProcess);
    intAttr.needsUpdate = true;

    this.geometry.setDrawRange(0, pointsToProcess);
    this.currentFrame = targetFrame;
  }

  update(delta: number): void {
    if (this.loadedBytes > this.parserState.lastParsedBytes) {
      advanceFrameParser(this.parserState, this.loadedBytes);
    }

    const totalLoadedFrames = this.parserState.frames.length;
    if (totalLoadedFrames === 0) return;

    let targetFrame: number;
    if (this.isPlaying && !this.audio.paused) {
      targetFrame = Math.floor(this.audio.currentTime * FPS) % totalLoadedFrames;
    } else {
      // Gentle animated breathing loop through available frames even before audio starts
      this.previewTime += delta;
      targetFrame = Math.floor(this.previewTime * FPS) % totalLoadedFrames;
    }

    if (targetFrame !== this.currentFrame) {
      this.renderFrame(targetFrame);
    }
  }
}
