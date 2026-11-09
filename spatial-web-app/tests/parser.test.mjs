import { describe, test } from 'node:test';
import assert from 'node:assert/strict';
import { advanceFrameParser, mulberry32 } from '../src/house-of-cards.ts';

// Build a synthetic quantized stream matching the HOCQ layout:
//   [0..3]     magic "HOCQ"
//   [4..7]     totalFrames u32 LE
//   per frame: numPoints u32 LE, 6 x float32 min/max X/Y/Z, then per point
//     u16 x, u16 y, u16 z, u8 intensity, 1 pad byte (8 bytes total)
function quantStream(totalFrames, frameCounts) {
  let size = 8;
  for (const n of frameCounts) size += 4 + 24 + n * 8;
  const buf = new ArrayBuffer(size);
  const dv = new DataView(buf);
  dv.setUint32(0, 0x51434f48, true); // "HOCQ" LE
  dv.setUint32(4, totalFrames, true);
  let o = 8;
  for (const n of frameCounts) {
    dv.setUint32(o, n, true);
    dv.setFloat32(o + 4, 0, true);
    dv.setFloat32(o + 8, 100, true);
    dv.setFloat32(o + 12, 0, true);
    dv.setFloat32(o + 16, 50, true);
    dv.setFloat32(o + 20, 0, true);
    dv.setFloat32(o + 24, 10, true);
    o += 28;
    for (let i = 0; i < n; i++) {
      dv.setUint16(o, i % 65536, true);
      dv.setUint16(o + 2, (i * 2) % 65536, true);
      dv.setUint16(o + 4, (i * 3) % 65536, true);
      dv.setUint8(o + 6, i % 256);
      dv.setUint8(o + 7, 0);
      o += 8;
    }
  }
  return buf;
}

function stateFromChunks(chunks, offsets, totalFrames) {
  return {
    chunks,
    chunkOffsets: offsets,
    initialized: true,
    isQuantized: true,
    totalFrames,
    nextByteOffset: 8,
    frames: [],
    parseError: null,
    scratchPool: [],
  };
}

describe('mulberry32', () => {
  test('same seed produces the same first value', () => {
    assert.equal(mulberry32(1337)(), mulberry32(1337)());
  });
  test('values stay in [0, 1)', () => {
    const rand = mulberry32(42);
    for (let i = 0; i < 100; i++) {
      const v = rand();
      assert.ok(v >= 0 && v < 1, `value ${v} out of range`);
    }
  });
});

describe('advanceFrameParser', () => {
  test('parses every frame from a single landed chunk', () => {
    const stream = quantStream(3, [10, 20, 15]);
    const state = stateFromChunks([stream], [0], 3);
    advanceFrameParser(state);
    assert.equal(state.parseError, null);
    assert.equal(state.frames.length, 3);
    assert.equal(state.frames[0].numPoints, 10);
    assert.equal(state.frames[1].numPoints, 20);
    assert.equal(state.frames[2].numPoints, 15);
    assert.ok(state.frames.every((f) => f.frameMinX === 0));
    assert.ok(state.frames.every((f) => f.rangeX === 100));
    assert.ok(state.frames.every((f) => f.rangeY === 50));
    assert.ok(state.frames.every((f) => f.rangeZ === 10));
  });

  test('invalid totalFrames latches a parse error', () => {
    const state = stateFromChunks([quantStream(0, [])], [0], 0);
    advanceFrameParser(state);
    assert.ok(state.parseError, 'expected parseError for totalFrames 0');
  });

  test('zero-point frame latches a parse error', () => {
    const state = stateFromChunks([quantStream(1, [0])], [0], 1);
    advanceFrameParser(state);
    assert.ok(state.parseError, 'expected parseError for numPoints 0');
    assert.equal(state.frames.length, 0);
  });

  test('frame straddling a chunk boundary parses via scratch', () => {
    const stream = quantStream(2, [4, 6]);
    const frame1Size = 28 + 4 * 8;
    const frame2Start = 8 + frame1Size;
    const cut = frame2Start + 10; // inside frame 2 header block
    const chunk0 = stream.slice(0, cut);
    const chunk1 = stream.slice(cut);
    const state = stateFromChunks([chunk0], [0], 2);
    // Only chunk 0 landed: frame 1 parses, frame 2 must wait.
    advanceFrameParser(state);
    assert.equal(state.frames.length, 1, 'straddled frame must wait');
    // Land chunk 1; the straddled frame must parse via a scratch copy.
    state.chunks = [chunk0, chunk1];
    state.chunkOffsets = [0, cut];
    advanceFrameParser(state);
    assert.equal(state.parseError, null);
    assert.equal(state.frames.length, 2);
    assert.equal(state.frames[1].numPoints, 6);
    assert.equal(state.frames[1].chunkIndex, -1, 'straddled frame uses scratch');
    assert.ok(state.scratchPool.length >= 1);
    assert.equal(state.frames[1].frameMinX, 0);
  });

  test('frame header u32 split across chunks waits then parses', () => {
    // Cut so frame 1's numPoints u32 itself straddles the boundary.
    const stream = quantStream(1, [5]);
    const cut = 8 + 2; // only 2 of the 4 numPoints bytes in chunk 0
    const chunk0 = stream.slice(0, cut);
    const chunk1 = stream.slice(cut);
    const state = stateFromChunks([chunk0], [0], 1);
    advanceFrameParser(state);
    assert.equal(state.frames.length, 0, 'split header must wait');
    state.chunks = [chunk0, chunk1];
    state.chunkOffsets = [0, cut];
    advanceFrameParser(state);
    assert.equal(state.parseError, null);
    assert.equal(state.frames.length, 1);
    assert.equal(state.frames[0].numPoints, 5);
  });
});
