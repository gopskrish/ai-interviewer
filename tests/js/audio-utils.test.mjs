import { test } from 'node:test';
import assert from 'node:assert/strict';
import { downsampleTo16k, int16ToBase64, base64ToInt16 } from '../../interviewer/web/static/audio-utils.mjs';

test('downsample 48k to 16k', () => assert.equal(downsampleTo16k(new Float32Array(4800), 48000).length, 1600));
test('downsample 44.1k to 16k', () => assert.equal(downsampleTo16k(new Float32Array(4410), 44100).length, 1600));
test('clips to int16', () => assert.deepEqual([...downsampleTo16k(new Float32Array([2, -2, 2]), 16000)], [32767, -32768, 32767]));
test('averages each group of input samples', () => assert.deepEqual([...downsampleTo16k(new Float32Array([0.5, 0.5, 0.5, -0.5, -0.5, -0.5]), 48000)], [16384, -16384]));
test('base64 roundtrip', () => assert.deepEqual([...base64ToInt16(int16ToBase64(new Int16Array([1, -2, 300])))], [1, -2, 300]));
test('base64 of PCM16 is little-endian', () => assert.equal(int16ToBase64(new Int16Array([1])), 'AQA='));
