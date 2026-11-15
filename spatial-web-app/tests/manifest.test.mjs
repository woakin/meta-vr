import { describe, test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const manifest = JSON.parse(readFileSync(join(root, 'public/data/chunks/manifest.json'), 'utf8'));

describe('chunk manifest consistency', () => {
  test('chunk sizes sum to totalSize', () => {
    const sum = manifest.chunks.reduce((a, c) => a + c.size, 0);
    assert.equal(sum, manifest.totalSize);
    assert.equal(manifest.chunks.length, manifest.chunkCount);
  });
  test('every chunk file exists with declared size', () => {
    for (const c of manifest.chunks) {
      const p = join(root, 'public', c.file.replace(/^\//, ''));
      assert.ok(existsSync(p), `missing ${c.file}`);
      assert.equal(readFileSync(p).byteLength, c.size, `size mismatch ${c.file}`);
    }
  });
  test('stream header is HOCQ with positive frame count', () => {
    const p = join(root, 'public', manifest.chunks[0].file.replace(/^\//, ''));
    const buf = readFileSync(p).subarray(0, 8);
    assert.equal(buf.subarray(0, 4).toString('ascii'), 'HOCQ');
    assert.ok(buf.readUInt32LE(4) > 0);
  });
});
