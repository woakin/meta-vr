import {
  AdditiveBlending,
  BufferAttribute,
  BufferGeometry,
  Color,
  createSystem,
  DistanceGrabbable,
  DynamicDrawUsage,
  Group,
  LocomotionEnvironment,
  Mesh,
  MeshBasicMaterial,
  MovementMode,
  PlaneGeometry,
  Points,
  RayInteractable,
  ShaderMaterial,
  SphereGeometry,
  UIKit,
  UIKitMLAsset,
  Vector3,
  VisibilityState,
} from '@iwsdk/core';

const FPS = 30;
const MAX_POINTS = 15000;
const INITIAL_PREVIEW_FRAME = 41;
const HOLOGRAM_SCALE = 0.0038;
const DEFAULT_POS = new Vector3(0, 1.42, -1.48);
// Desktop fly speed (m/s) for E/C vertical glide. Keys avoid Q on purpose:
// in the XR emulator Q is the left controller trigger, so holding it would
// distance-grab the hologram instead of sinking the camera. Driven through
// world.setAuthoredPlayerPosition so the locomotor worker (which owns the rig
// pose and would overwrite direct edits) stays authoritative. An invisible
// kinematic hover platform follows at the chosen altitude, so the locomotor
// remains grounded there instead of gravity pulling it back down. There is
// no floor in the scene — the bounds only keep the player recoverably lost.
const FLY_SPEED = 1.6;
const MIN_FLY_Y = -30;
const MAX_FLY_Y = 30;
// Intro assembly: each point flies in from a scattered position with a
// per-point stagger, then eases into its decoded place.
const ASSEMBLE_SPAN = 2.4;
const ASSEMBLE_STAGGER = 2.8;
const CHUNK_CONCURRENCY = 4;

function dataUrl(path: string): string {
  return `${import.meta.env.BASE_URL}${path.replace(/^\/+/u, '')}`;
}

// Match a fly key by physical position (code) or by printed label (key).
// code alone misbinds on layouts that move the letter: on AZERTY the key
// labeled Q reports code KeyA, so labeled Q strafed instead of descending.
function isFlyKey(e: KeyboardEvent, code: string, key: string): boolean {
  if (e.ctrlKey || e.altKey || e.metaKey) return false;
  return e.code === code || e.key.toLowerCase() === key;
}

function isActionKey(e: KeyboardEvent, code: string, key: string): boolean {
  return isFlyKey(e, code, key);
}

// Deterministic PRNG (mulberry32) so intro scatter is identical in the
// runtime and editor realms and across resets.
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// One parsed frame. Point data lives either inside a single landed chunk
// (chunkIndex/offsetInChunk address it with the chunk's own typed-array
// views) or in a small scratch copy (frames straddling a chunk boundary,
// or chunk-local frames whose offset would misalign a u16/float view).
interface FrameData {
  numPoints: number;
  /** Byte offset of the point-data start inside its source buffer. */
  dataOffset: number;
  /** Index into ParserState.chunks, or -1 when reading from `scratch`. */
  chunkIndex: number;
  /** Offset of the point data inside the scratch copy, when straddled. */
  scratch: ArrayBuffer | null;
  floatOffset?: number;
  frameMinX?: number;
  rangeX?: number;
  frameMinY?: number;
  rangeY?: number;
  frameMinZ?: number;
  rangeZ?: number;
}

const HEADER_BYTES = 28; // quantized frame header

export interface ParserState {
  /** Landed chunk buffers by manifest index; null = not yet arrived. */
  chunks: (ArrayBuffer | null)[];
  /** Virtual (concatenated-stream) start offset of each chunk. */
  chunkOffsets: number[];
  /** Virtual start offset of the chunk that holds frame point data. */
  initialized: boolean;
  isQuantized: boolean;
  totalFrames: number;
  nextByteOffset: number;
  frames: FrameData[];
  /** Latched parse error; parsing stops when set. */
  parseError: string | null;
  /** Scratch buffers retained for straddled/misaligned frames. */
  scratchPool: ArrayBuffer[];
}

// Locate which landed chunk contains a virtual stream offset, or -1.
function locateChunk(state: ParserState, virtualOffset: number): number {
  const offsets = state.chunkOffsets;
  let lo = 0;
  let hi = offsets.length - 1;
  let found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (offsets[mid] <= virtualOffset) {
      found = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  if (found < 0) return -1;
  const chunk = state.chunks[found];
  if (!chunk) return -1;
  return virtualOffset - offsets[found] < chunk.byteLength ? found : -1;
}

// Copy [virtualStart, virtualStart+length) into a small contiguous buffer.
// Returns null when any byte of the range has not landed yet.
function copyRange(
  state: ParserState,
  virtualStart: number,
  length: number,
): ArrayBuffer | null {
  const out = new ArrayBuffer(length);
  const dst = new Uint8Array(out);
  for (let i = 0; i < length; i++) {
    const virtual = virtualStart + i;
    const chunkIndex = locateChunk(state, virtual);
    if (chunkIndex < 0) return null;
    const src = state.chunks[chunkIndex]!;
    dst[i] = new Uint8Array(src)[virtual - state.chunkOffsets[chunkIndex]];
  }
  return out;
}

// Read a little-endian u32 at a virtual offset; null when not landed.
function readU32(state: ParserState, virtualOffset: number): number | null {
  let value = 0;
  for (let i = 0; i < 4; i++) {
    const chunkIndex = locateChunk(state, virtualOffset + i);
    if (chunkIndex < 0) return null;
    const src = state.chunks[chunkIndex]!;
    value +=
      new Uint8Array(src)[virtualOffset + i - state.chunkOffsets[chunkIndex]] *
      2 ** (8 * i);
  }
  return value;
}

export function advanceFrameParser(state: ParserState): void {
  if (state.parseError || !state.initialized) return;
  const totalFrames = state.totalFrames;
  if (!Number.isInteger(totalFrames) || totalFrames <= 0) {
    state.parseError = `Invalid totalFrames in stream header: ${totalFrames}`;
    return;
  }

  while (state.frames.length < totalFrames) {
    const frameStart = state.nextByteOffset;
    const numPoints = readU32(state, frameStart);
    if (numPoints === null) return; // header bytes have not landed yet
    if (numPoints === 0 || numPoints > MAX_POINTS * 2) {
      state.parseError =
        `Invalid numPoints ${numPoints} at stream offset ${frameStart}`;
      return;
    }

    const headerSize = state.isQuantized ? HEADER_BYTES : 4;
    const frameSize = headerSize + numPoints * (state.isQuantized ? 8 : 16);
    const frameEnd = frameStart + frameSize;

    // Point-data start, as a chunk-local offset plus the source chunk.
    // The whole frame (header included) must share one chunk, otherwise
    // header reads would run off the chunk that holds the body.
    const dataVirtual = frameStart + headerSize;
    const headerChunkIndex = locateChunk(state, frameStart);
    const chunkIndex = locateChunk(state, dataVirtual);
    const dataChunk = chunkIndex >= 0 ? state.chunks[chunkIndex] : null;
    const endChunkIndex = locateChunk(state, frameEnd - 1);
    const straddled =
      headerChunkIndex < 0 ||
      chunkIndex < 0 ||
      endChunkIndex !== chunkIndex ||
      chunkIndex !== headerChunkIndex ||
      !dataChunk;
    const misaligned =
      !straddled &&
      (state.isQuantized
        ? (dataVirtual - state.chunkOffsets[chunkIndex]) % 2 !== 0
        : (dataVirtual - state.chunkOffsets[chunkIndex]) % 4 !== 0);

    if (straddled || misaligned) {
      const scratch = copyRange(state, frameStart, frameSize);
      if (scratch === null) return; // frame bytes have not landed yet
      state.scratchPool.push(scratch);
      const scratchView = new DataView(scratch);
      if (state.isQuantized) {
        const h = 4;
        state.frames.push({
          numPoints,
          dataOffset: headerSize,
          chunkIndex: -1,
          scratch,
          frameMinX: scratchView.getFloat32(h, true),
          rangeX:
            scratchView.getFloat32(h + 4, true) -
            scratchView.getFloat32(h, true) ||
            1,
          frameMinY: scratchView.getFloat32(h + 8, true),
          rangeY:
            scratchView.getFloat32(h + 12, true) -
            scratchView.getFloat32(h + 8, true) ||
            1,
          frameMinZ: scratchView.getFloat32(h + 16, true),
          rangeZ:
            scratchView.getFloat32(h + 20, true) -
            scratchView.getFloat32(h + 16, true) ||
            1,
        });
      } else {
        state.frames.push({
          numPoints,
          dataOffset: headerSize,
          chunkIndex: -1,
          scratch,
          floatOffset: headerSize / 4,
        });
      }
    } else {
      const view = new DataView(state.chunks[chunkIndex]!);
      const dataOffset = dataVirtual - state.chunkOffsets[chunkIndex];
      if (state.isQuantized) {
        const h = dataOffset - HEADER_BYTES + 4;
        state.frames.push({
          numPoints,
          dataOffset,
          chunkIndex,
          scratch: null,
          frameMinX: view.getFloat32(h, true),
          rangeX: view.getFloat32(h + 4, true) - view.getFloat32(h, true) || 1,
          frameMinY: view.getFloat32(h + 8, true),
          rangeY:
            view.getFloat32(h + 12, true) - view.getFloat32(h + 8, true) || 1,
          frameMinZ: view.getFloat32(h + 16, true),
          rangeZ:
            view.getFloat32(h + 20, true) - view.getFloat32(h + 16, true) || 1,
        });
      } else {
        state.frames.push({
          numPoints,
          dataOffset,
          chunkIndex,
          scratch: null,
          floatOffset: dataOffset / 4,
        });
      }
    }
    state.nextByteOffset = frameEnd;
  }
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
    chunks: [],
    chunkOffsets: [],
    initialized: false,
    isQuantized: false,
    totalFrames: 0,
    nextByteOffset: 0,
    frames: [],
    parseError: null,
    scratchPool: [],
  };

  private abortController = new AbortController();
  private platformGeometry: BufferGeometry | null = null;
  private platformMaterial: MeshBasicMaterial | null = null;
  private hitGeometry: BufferGeometry | null = null;
  private hitMaterial: MeshBasicMaterial | null = null;
  private assemblySeed = 1337;
  private assemblyRandom = mulberry32(this.assemblySeed);

  private geometry!: BufferGeometry;
  private posAttr!: BufferAttribute;
  private intAttr!: BufferAttribute;
  private shaderMaterial!: ShaderMaterial;
  private hologramGroup!: Group;
  private presenting: boolean | null = null;

  private statusLabel: UIKit.Text | null = null;
  private playButtonLabel: UIKit.Text | null = null;

  private flyUp = false;
  private flyDown = false;
  private flyTarget = new Vector3();
  private hoverPlatform!: Group;
  private flyAltitude = 0;

  // Intro assembly state. basePositions holds the pristine decoded frame
  // (the live attribute is the blended display copy); scatterPos holds each
  // point's far starting position. Both allocated once, never in update().
  private introStarted = false;
  private introActive = false;
  private introTime = 0;
  private baseCount = 0;
  private basePositions = new Float32Array(MAX_POINTS * 3);
  private scatterPos = new Float32Array(MAX_POINTS * 3);

  // Auto-hiding control deck.
  private panelAsset: UIKitMLAsset | null = null;
  private hideTimer = 0;
  private camPos = new Vector3();
  private camDir = new Vector3();
  private toPanel = new Vector3();

  init(): void {
    // 1. Set up audio element
    this.audio = new Audio(dataUrl('data/HouseOfCards_DataSample.mp3'));
    this.audio.loop = true;
    // Keep the initial network budget for the LiDAR stream; the MP3 buffers
    // when the user presses play.
    this.audio.preload = 'metadata';

    // 2. Build Three.js Point Cloud geometry + custom VR ShaderMaterial
    const initialPositions = new Float32Array(MAX_POINTS * 3);
    const initialIntensities = new Float32Array(MAX_POINTS);

    this.geometry = new BufferGeometry();
    this.posAttr = new BufferAttribute(initialPositions, 3);
    this.posAttr.setUsage(DynamicDrawUsage);
    this.intAttr = new BufferAttribute(initialIntensities, 1);
    this.intAttr.setUsage(DynamicDrawUsage);

    this.geometry.setAttribute('position', this.posAttr);
    this.geometry.setAttribute('intensity', this.intAttr);

    this.shaderMaterial = new ShaderMaterial({
      uniforms: {
        colorA: { value: new Color('#4facfe') },
        colorB: { value: new Color('#00f2fe') },
        colorC: { value: new Color('#f093fb') },
        pointSize: { value: 3.4 },
        uBrightness: { value: 1.0 },
      },
      vertexShader: `
        attribute float intensity;
        varying float vIntensity;
        varying float vZ;
        uniform float pointSize;
        uniform float uBrightness;
        varying float vBrightness;

        void main() {
          vIntensity = intensity;
          vZ = position.z;
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          // gl_PointSize is in DEVICE pixels. In XR the drawing buffer packs
          // far more pixels per degree than the flat preview, so raw pixel
          // sizes look huge and blobby in-headset. We size points as screen
          // fractions instead: projectionMatrix[1][1] scales with the
          // viewport height, so 0.5 / P11 normalizes it — and XR shrinks
          // points further via uBrightness (set in update() when presenting).
          float heightPx = 0.5 / projectionMatrix[1][1];
          float sizeScale = pointSize * (2.6 / max(0.25, -mvPosition.z)) * heightPx;
          gl_PointSize = clamp(sizeScale, 1.0, 10.0);
          vBrightness = uBrightness;
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        uniform vec3 colorA;
        uniform vec3 colorB;
        uniform vec3 colorC;

        varying float vIntensity;
        varying float vZ;
        varying float vBrightness;

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
          // The tone curve defuses the additive white core when dense points
          // overlap, preserving the bust's stripes in XR instead of blowing
          // them to solid white.
          vec3 display = mixColor * (normalizedInt * 0.85 + 0.45) * vBrightness;
          gl_FragColor = vec4(display / (display + 0.75), alpha);
        }
      `,
      transparent: true,
      depthWrite: false,
      blending: AdditiveBlending,
    });

    const points = new Points(this.geometry, this.shaderMaterial);
    // Draw range changes every frame, so a static bounding sphere would
    // cull incorrectly. Skip culling rather than recomputing bounds.
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
    this.hologramGroup = new Group();
    this.hologramGroup.name = 'HouseOfCardsHologram';

    // Invisible hit volume so Gaze + Pinch raycasting can easily target the point cloud
    const hitGeo = new SphereGeometry(0.42, 16, 16);
    const hitMat = new MeshBasicMaterial({
      transparent: true,
      opacity: 0.0,
      depthWrite: false,
    });
    this.hitGeometry = hitGeo;
    this.hitMaterial = hitMat;
    const hitMesh = new Mesh(hitGeo, hitMat);
    this.hologramGroup.add(hitMesh);

    this.hologramGroup.add(points);

    const entity = this.world.createTransformEntity(this.hologramGroup);
    entity.object3D!.position.copy(DEFAULT_POS);
    entity
      .addComponent(RayInteractable)
      .addComponent(DistanceGrabbable, {
        movementMode: MovementMode.MoveFromTarget,
        returnToOrigin: false,
      });

    // Invisible kinematic floor for free-flight. It follows the player at
    // flyAltitude, giving the locomotor a grounded surface at any height;
    // without it, gravity in the locomotion worker pulls E/C flight back
    // down. Large enough that WASD never reaches an edge
    // before the platform recenters on the next frame.
    this.platformGeometry = new PlaneGeometry(240, 240);
    this.platformMaterial = new MeshBasicMaterial();
    const platformMesh = new Mesh(
      this.platformGeometry,
      this.platformMaterial,
    );
    platformMesh.rotation.x = -Math.PI / 2;
    platformMesh.visible = false;
    this.hoverPlatform = new Group();
    this.hoverPlatform.name = 'HoverPlatform';
    this.hoverPlatform.add(platformMesh);
    this.flyAltitude = this.player.position.y;
    this.hoverPlatform.position.set(
      this.player.position.x,
      this.flyAltitude,
      this.player.position.z,
    );
    const platformEntity = this.world.createTransformEntity(this.hoverPlatform);
    platformEntity.addComponent(LocomotionEnvironment, {
      type: 'kinematic',
    });

    // 4. Wire Spatial UI Panel controls & keyboard shortcuts
    this.wirePanelControls();
    this.wireKeyboardControls();

    // 5. Start streaming the quantized LiDAR chunks
    this.cleanupFuncs.push(() => {
      this.abortController.abort();
      this.geometry?.dispose();
      this.shaderMaterial?.dispose();
      this.hitGeometry?.dispose();
      this.hitMaterial?.dispose();
      this.platformGeometry?.dispose();
      this.platformMaterial?.dispose();
      if (!this.audio.paused) this.audio.pause();
    });
    void this.loadChunkStream();
  }

  private wirePanelControls(): void {
    const panel = this.world.getSceneObject<UIKitMLAsset>('welcome-panel');
    if (!panel) {
      console.error(
        '[HouseOfCards] welcome-panel scene object not found; panel controls disabled.',
      );
      return;
    }
    this.panelAsset = panel;

    this.statusLabel = panel.getElementById<UIKit.Text>('status-text') ?? null;
    this.playButtonLabel =
      panel.getElementById<UIKit.Text>('play-label') ?? null;

    const playBtn = panel.getElementById('play-button');
    const vibeBtn = panel.getElementById('vibe-button');
    const resetBtn = panel.getElementById('reset-button');
    const xrButton = panel.getElementById('xr-button');
    const exitButton = panel.getElementById('exit-button');

    const onPlay = () => this.togglePlay();
    const onVibe = () => this.randomizeVibe();
    const onReset = () => this.resetHologram();

    playBtn?.addEventListener('click', onPlay);
    vibeBtn?.addEventListener('click', onVibe);
    resetBtn?.addEventListener('click', onReset);

    // XR entry/exit (merged from the starter PanelSystem so this
    // stays a single-system app).
    if (!this.world.xrEnabled) {
      xrButton?.setProperties({ display: 'none' });
      exitButton?.setProperties({ display: 'none' });
    } else {
      const launchXR = () => this.world.launchXR();
      const exitXR = () => this.world.exitXR();
      xrButton?.addEventListener('click', launchXR);
      exitButton?.addEventListener('click', exitXR);
      this.cleanupFuncs.push(
        () => xrButton?.removeEventListener('click', launchXR),
        () => exitButton?.removeEventListener('click', exitXR),
      );
    }

    this.cleanupFuncs.push(
      () => playBtn?.removeEventListener('click', onPlay),
      () => vibeBtn?.removeEventListener('click', onVibe),
      () => resetBtn?.removeEventListener('click', onReset),
    );
    const unsubscribeVisibility = this.world.visibilityState.subscribe(
      (state) => {
        const is2D = state === VisibilityState.NonImmersive;
        xrButton?.setProperties({ display: is2D ? 'flex' : 'none' });
        exitButton?.setProperties({ display: is2D ? 'none' : 'flex' });
        if (state === VisibilityState.VisibleBlurred && this.isPlaying) {
          this.audio.pause();
          this.isPlaying = false;
          this.updatePlayLabel();
        }
      },
    );
    this.cleanupFuncs.push(unsubscribeVisibility);
  }

  private wireKeyboardControls(): void {
    const onKeyDown = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }
      if (isActionKey(e, 'KeyP', 'p')) {
        if (e.repeat) return;
        e.preventDefault();
        this.togglePlay();
      } else if (isActionKey(e, 'KeyV', 'v')) {
        if (e.repeat) return;
        e.preventDefault();
        this.randomizeVibe();
      } else if (isActionKey(e, 'KeyR', 'r')) {
        if (e.repeat) return;
        e.preventDefault();
        this.resetHologram();
      } else if (isFlyKey(e, 'KeyE', 'e')) {
        this.flyUp = true;
      } else if (isFlyKey(e, 'KeyC', 'c')) {
        this.flyDown = true;
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (isFlyKey(e, 'KeyE', 'e')) {
        this.flyUp = false;
      } else if (isFlyKey(e, 'KeyC', 'c')) {
        this.flyDown = false;
      }
    };
    // Avoid a stuck climb/dive if the key is released outside the window.
    const onBlur = () => {
      this.flyUp = false;
      this.flyDown = false;
    };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', onBlur);
    this.cleanupFuncs.push(
      () => window.removeEventListener('keydown', onKeyDown),
      () => window.removeEventListener('keyup', onKeyUp),
      () => window.removeEventListener('blur', onBlur),
    );
  }

  private updatePlayLabel(): void {
    this.playButtonLabel?.setProperties({
      text: this.isPlaying ? 'PAUSE AUDIO & SYNC' : 'PLAY RADIOHEAD SYNC (P)',
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
          this.isPlaying = false;
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
  }

  public resetHologram(): void {
    this.hologramGroup.position.copy(DEFAULT_POS);
    this.hologramGroup.rotation.set(0, 0, 0);
    // Burst apart and reassemble — doubles as an intro replay. Re-decode
    // the current frame so the assembly targets a pristine decoded
    // position, never a mid-blend position.
    if (this.currentFrame >= 0) {
      const frame = this.parserState.frames[this.currentFrame];
      if (frame) {
        const positions = this.posAttr.array as Float32Array;
        const intensities = this.intAttr.array as Float32Array;
        const count = this.decodeFrameIntoAttributes(
          frame,
          positions,
          intensities,
        );
        if (count > 0) {
          this.basePositions.set(positions.subarray(0, count * 3));
          this.baseCount = count;
        }
      }
    }
    if (this.baseCount > 0) {
      this.startAssembly();
      this.updateStatus('Assembling hologram...');
    } else {
      this.updateStatus('No decoded frame yet - waiting for LiDAR stream...');
    }
  }

  // Initialize from landed chunk 0 only. Later chunks are stored per-index;
  // the parser addresses the concatenated stream without a 205 MB buffer.
  private initParserBuffer(chunk0: ArrayBuffer): void {
    if (chunk0.byteLength < 8) {
      throw new Error(
        `Stream header truncated: chunk 0 has ${chunk0.byteLength} bytes`,
      );
    }
    const dataview = new DataView(chunk0);
    const magicView = new Uint8Array(chunk0, 0, 4);
    const isQuant =
      magicView[0] === 0x48 &&
      magicView[1] === 0x4f &&
      magicView[2] === 0x43 &&
      magicView[3] === 0x51; // 'HOCQ'

    const totalFrames = isQuant
      ? dataview.getUint32(4, true)
      : dataview.getUint32(0, true);
    if (!Number.isInteger(totalFrames) || totalFrames <= 0) {
      throw new Error(`Invalid totalFrames in stream header: ${totalFrames}`);
    }

    this.parserState = {
      chunks: [chunk0],
      chunkOffsets: [0],
      initialized: true,
      isQuantized: isQuant,
      totalFrames,
      nextByteOffset: isQuant ? 8 : 4,
      frames: [],
      parseError: null,
      scratchPool: [],
    };
  }

  private async fetchChunk(
    chunk: { file: string; size?: number },
    index: number,
  ): Promise<ArrayBuffer> {
    const res = await fetch(dataUrl(chunk.file), {
      signal: this.abortController.signal,
    });
    if (!res.ok) {
      throw new Error(`Chunk ${index} failed: HTTP ${res.status}`);
    }
    const buf = await res.arrayBuffer();
    if (chunk.size !== undefined && buf.byteLength !== chunk.size) {
      throw new Error(
        `Chunk ${index} size mismatch: ${buf.byteLength} != ${chunk.size}`,
      );
    }
    return buf;
  }

  private async loadChunkStream(): Promise<void> {
    try {
      this.updateStatus('Loading LiDAR manifest...');
      const manifestRes = await fetch(dataUrl('data/chunks/manifest.json'));
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
      if (chunkList.some((chunk) => !chunk.size || chunk.size <= 0)) {
        throw new Error('Manifest chunks must declare positive sizes');
      }

      const offsets = new Array<number>(chunkList.length);
      let runningOffset = 0;
      for (let i = 0; i < chunkList.length; i++) {
        offsets[i] = runningOffset;
        runningOffset += chunkList[i].size!;
      }
      if (runningOffset !== totalSize) {
        throw new Error(
          `Manifest size mismatch: chunks ${runningOffset} != total ${totalSize}`,
        );
      }

      // Load chunk 0 immediately for instant preview.
      const chunk0Buf = await this.fetchChunk(chunkList[0], 0);

      this.initParserBuffer(chunk0Buf);
      this.loadedBytes = chunk0Buf.byteLength;
      advanceFrameParser(this.parserState);
      this.renderFrame(
        Math.min(INITIAL_PREVIEW_FRAME, this.parserState.frames.length - 1),
      );
      this.updateStatus(
        `Hologram Ready (${this.parserState.frames.length} frames) - Streaming...`,
      );

      if (chunkList.length === 1) {
        this.updateStatus(`All ${this.parserState.frames.length} LiDAR Frames Ready`);
        return;
      }

      // Fetch the rest with bounded concurrency. The parser still receives a
      // contiguous prefix: out-of-order completions wait in `combined` until
      // every earlier chunk has landed.
      const completed = new Array<boolean>(chunkList.length).fill(false);
      completed[0] = true;
      let nextIndex = 1;
      let contiguousIndex = 1;
      let contiguousBytes = chunk0Buf.byteLength;
      let downloadedBytes = chunk0Buf.byteLength;
      let lastReportedPct = -1;
      let failed = false;

      const reportProgress = (force = false) => {
        if (failed) return;
        const pct = Math.min(100, Math.round((downloadedBytes / totalSize) * 100));
        if (!force && pct === lastReportedPct) return;
        lastReportedPct = pct;
        advanceFrameParser(this.parserState);
        if (this.parserState.parseError) {
          this.updateStatus(`LiDAR Parse Error: ${this.parserState.parseError}`);
          return;
        }
        this.updateStatus(
          pct === 100
            ? `All ${this.parserState.frames.length} LiDAR Frames Ready`
            : `Streaming LiDAR (${pct}% - ${this.parserState.frames.length} frames)`,
        );
      };

      const worker = async (): Promise<void> => {
        while (!failed && nextIndex < chunkList.length) {
          const index = nextIndex;
          nextIndex += 1;
          let buf: ArrayBuffer;
          try {
            buf = await this.fetchChunk(chunkList[index], index);
          } catch (err) {
            failed = true;
            throw err;
          }
          // Store per-index instead of copying into a contiguous buffer.
          if (!this.parserState.chunks[index]) {
            this.parserState.chunks[index] = buf;
            this.parserState.chunkOffsets[index] = offsets[index];
          }
          completed[index] = true;
          downloadedBytes += buf.byteLength;

          while (
            contiguousIndex < chunkList.length &&
            completed[contiguousIndex]
          ) {
            contiguousBytes =
              offsets[contiguousIndex] + chunkList[contiguousIndex].size!;
            contiguousIndex += 1;
          }
          this.loadedBytes = contiguousBytes;
          reportProgress(index === chunkList.length - 1);
        }
      };

      await Promise.all(
        Array.from(
          { length: Math.min(CHUNK_CONCURRENCY, chunkList.length - 1) },
          () => worker(),
        ),
      );
      this.loadedBytes = totalSize;
      advanceFrameParser(this.parserState);
      reportProgress(true);
    } catch (err) {
      if (this.abortController.signal.aborted) return;
      console.error('Failed to stream House of Cards LiDAR chunks:', err);
      this.updateStatus('LiDAR Load Error - check /public/data');
    }
  }

  private decodeFrameIntoAttributes(
    frame: FrameData,
    positions: Float32Array,
    intensities: Float32Array,
  ): number {
    const pointsToProcess = Math.min(frame.numPoints, MAX_POINTS);
    const source =
      frame.chunkIndex >= 0
        ? this.parserState.chunks[frame.chunkIndex]
        : frame.scratch;
    if (!source) return 0;
    const u8View = new Uint8Array(source);
    const u16View = new Uint16Array(source);
    const floatData = new Float32Array(source);

    if (
      this.parserState.isQuantized &&
      frame.frameMinX !== undefined &&
      u16View &&
      u8View
    ) {
      const frameMinX = frame.frameMinX;
      const frameMinY = frame.frameMinY!;
      const frameMinZ = frame.frameMinZ!;
      const scaleX = frame.rangeX! / 65535;
      const scaleY = frame.rangeY! / 65535;
      const scaleZ = frame.rangeZ! / 65535;

      let pOffset = frame.dataOffset + 6;
      let u16Idx = frame.dataOffset >> 1;
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
    return pointsToProcess;
  }

  private renderFrame(targetFrame: number): void {
    const frame = this.parserState.frames[targetFrame];
    if (!frame) return;

    const positions = this.posAttr.array as Float32Array;
    const intensities = this.intAttr.array as Float32Array;
    const pointsToProcess = this.decodeFrameIntoAttributes(
      frame,
      positions,
      intensities,
    );
    if (pointsToProcess <= 0) return;

    // Single upload pass. When an assembly is active the same range is
    // blended in place right after; flagging here and again in updateAssembly
    // would upload twice per frame.
    this.posAttr.clearUpdateRanges();
    this.posAttr.addUpdateRange(0, pointsToProcess * 3);
    this.posAttr.needsUpdate = true;

    this.intAttr.clearUpdateRanges();
    this.intAttr.addUpdateRange(0, pointsToProcess);
    this.intAttr.needsUpdate = true;

    this.geometry.setDrawRange(0, pointsToProcess);
    this.currentFrame = targetFrame;

    // Snapshot the pristine frame only when an assembly can consume it.
    // Copying 45k floats on every normal playback frame is wasted bandwidth.
    // While assembling, the frame is held (update() skips advance), so the
    // base stays frozen and points ease to a stable target.
    if (!this.introStarted) {
      this.basePositions.set(positions.subarray(0, pointsToProcess * 3));
      this.baseCount = pointsToProcess;
      this.introStarted = true;
      this.startAssembly();
      this.updateStatus('Assembling hologram...');
    }
  }

  /** Scatter every point into the void and restart the fly-in. */
  private startAssembly(): void {
    this.assemblyRandom = mulberry32(this.assemblySeed);
    // Scatter the full capacity, not just the first frame's count: denser
    // frames decoded during streaming must not inherit zeroed starts.
    for (let i = 0; i < MAX_POINTS; i += 1) {
      // Random unit vector via spherical sampling.
      const theta = this.assemblyRandom() * Math.PI * 2;
      const z = this.assemblyRandom() * 2 - 1;
      const r = Math.sqrt(Math.max(0, 1 - z * z));
      const dist = 120 + this.assemblyRandom() * 260;
      const o = i * 3;
      this.scatterPos[o] = this.basePositions[o] + r * Math.cos(theta) * dist;
      this.scatterPos[o + 1] =
        this.basePositions[o + 1] + r * Math.sin(theta) * dist;
      this.scatterPos[o + 2] = this.basePositions[o + 2] + z * dist;
    }
    this.introTime = 0;
    this.introActive = true;
  }

  /** Blend scattered starts into decoded targets; call once per frame. */
  private updateAssembly(delta: number): void {
    this.introTime += delta;
    const positions = this.posAttr.array as Float32Array;
    const count = this.baseCount;
    let done = true;
    for (let i = 0; i < count; i += 1) {
      const t =
        (this.introTime - (i / count) * ASSEMBLE_STAGGER) / ASSEMBLE_SPAN;
      if (t < 1) done = false;
      const clamped = t <= 0 ? 0 : t >= 1 ? 1 : t;
      const e = 1 - (1 - clamped) * (1 - clamped) * (1 - clamped);
      const o = i * 3;
      positions[o] =
        this.scatterPos[o] + (this.basePositions[o] - this.scatterPos[o]) * e;
      positions[o + 1] =
        this.scatterPos[o + 1] +
        (this.basePositions[o + 1] - this.scatterPos[o + 1]) * e;
      positions[o + 2] =
        this.scatterPos[o + 2] +
        (this.basePositions[o + 2] - this.scatterPos[o + 2]) * e;
    }
    this.posAttr.clearUpdateRanges();
    this.posAttr.addUpdateRange(0, count * 3);
    this.posAttr.needsUpdate = true;
    if (done) {
      this.introActive = false;
      // Land exactly on the decoded frame (no residual float error).
      positions.set(this.basePositions.subarray(0, count * 3));
      this.posAttr.clearUpdateRanges();
      this.posAttr.addUpdateRange(0, count * 3);
      this.posAttr.needsUpdate = true;
    }
  }

  update(delta: number): void {
    advanceFrameParser(this.parserState);

    const totalLoadedFrames = this.parserState.frames.length;
    if (totalLoadedFrames === 0) return;
    if (this.parserState.parseError) return;

    // Hold the frame while assembling so the blend target stays stable and
    // the decode upload is not duplicated every frame.
    if (!this.introActive || this.currentFrame < 0) {
      let targetFrame: number;
      if (this.isPlaying && !this.audio.paused) {
        // Sync against the full timeline, not the still-streaming subset;
        // otherwise the effective playback rate changes as chunks land.
        const timelineFrames =
          this.parserState.totalFrames || totalLoadedFrames;
        if (!Number.isFinite(timelineFrames) || timelineFrames <= 0) {
          targetFrame = Math.floor(this.previewTime * FPS) % totalLoadedFrames;
        } else {
          targetFrame =
            Math.floor(this.audio.currentTime * FPS) % timelineFrames;
        }
        if (
          !Number.isFinite(targetFrame) ||
          targetFrame < 0 ||
          targetFrame >= totalLoadedFrames
        ) {
          targetFrame = totalLoadedFrames - 1;
        }
      } else {
        // Gentle animated breathing loop through available frames even before audio starts
        this.previewTime += delta;
        targetFrame =
          Math.floor(this.previewTime * FPS) % totalLoadedFrames;
      }

      if (targetFrame !== this.currentFrame) {
        this.renderFrame(targetFrame);
      }
    }

    // Fly-in assembly runs until every point has eased home.
    if (this.introActive) {
      this.updateAssembly(delta);
    }

    // XR presenting: smaller, softer points. The flat preview is ~900px tall;
    // an XR eye is ~2000px+ per eye, so without this the additive points
    // overlap into a blown-out cloud and lose the bust's fine structure.
    // Only write uniforms when the presenting mode actually flips.
    const presenting =
      this.world.visibilityState.peek() !== VisibilityState.NonImmersive;
    if (presenting !== this.presenting) {
      this.presenting = presenting;
      this.shaderMaterial.uniforms.uBrightness.value = presenting ? 1.02 : 1.0;
      this.shaderMaterial.uniforms.pointSize.value = presenting ? 2.1 : 3.4;
    }

    // Auto-hiding control deck (desktop only — the XR exit button must stay
    // reachable in-headset). While the track plays, the card hides after a
    // few seconds and returns the moment you look its way or pause.
    if (this.panelAsset) {
      const immersive = presenting;
      if (!this.isPlaying || immersive) {
        this.panelAsset.visible = true;
        this.hideTimer = 0;
      } else {
        this.world.camera.getWorldPosition(this.camPos);
        this.world.camera.getWorldDirection(this.camDir);
        this.panelAsset.getWorldPosition(this.toPanel);
        this.toPanel.sub(this.camPos).normalize();
        if (this.camDir.dot(this.toPanel) > 0.8) {
          this.panelAsset.visible = true;
          this.hideTimer = 0;
        } else {
          this.hideTimer += delta;
          if (this.hideTimer > 5) this.panelAsset.visible = false;
        }
      }
    }

    // Desktop free-flight: E rises, C sinks. Routed through the
    // authored-placement channel so the locomotor worker teleports its
    // internal pose instead of fighting a direct rig edit. The hover
    // platform then holds that altitude, so releasing the key leaves the
    // player floating — WASD slides along the platform at that height and
    // gravity never pulls the rig back down.
    const flyDir = (this.flyUp ? 1 : 0) - (this.flyDown ? 1 : 0);
    if (flyDir !== 0) {
      // Integrate from flyAltitude, not player.position.y: the rig pose is
      // lerped back from the locomotion worker and lags the authored target.
      // Rising stayed exact because the platform carries the rig up, but
      // descending had to fall after the platform, so Q sank ~15% slower
      // than E rose (and stalled further on slow frames).
      if (this.player.position.y > this.flyAltitude + 0.5) {
        // Something else placed the player well above the platform (e.g. an
        // XR teleport); adopt that height instead of snapping back down.
        this.flyAltitude = this.player.position.y;
      }
      const newY = Math.min(
        MAX_FLY_Y,
        Math.max(MIN_FLY_Y, this.flyAltitude + flyDir * FLY_SPEED * delta),
      );
      if (newY !== this.flyAltitude) {
        this.flyAltitude = newY;
        this.flyTarget.copy(this.player.position);
        this.flyTarget.y = newY;
        this.world.setAuthoredPlayerPosition(this.flyTarget);
      }
    } else if (this.player.position.y < this.flyAltitude - 0.05) {
      // Something else placed the player lower (e.g. an XR teleport).
      // Resync so the platform does not hover uselessly overhead.
      this.flyAltitude = Math.max(MIN_FLY_Y, this.player.position.y);
    }
    this.hoverPlatform.position.set(
      this.player.position.x,
      this.flyAltitude,
      this.player.position.z,
    );
  }
}
