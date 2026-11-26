// PCM helpers for the practice page (and its tests): browser audio is Float32 at the
// AudioContext rate; the interviewer speaks PCM16 mono, 16 kHz in and 24 kHz out.

export const TARGET_RATE = 16000;

function toInt16(sample) {
  const s = Math.max(-1, Math.min(1, sample));
  return s < 0 ? Math.round(s * 0x8000) : Math.round(s * 0x7fff);
}

// Averages each group of input samples (a simple low-pass) and clips to int16.
export function downsampleTo16k(float32, inRate) {
  const length = Math.floor((float32.length * TARGET_RATE) / inRate);
  const out = new Int16Array(length);
  for (let i = 0; i < length; i++) {
    const start = Math.floor((i * inRate) / TARGET_RATE);
    const end = Math.max(start + 1, Math.min(float32.length, Math.floor(((i + 1) * inRate) / TARGET_RATE)));
    let sum = 0;
    for (let j = start; j < end; j++) sum += float32[j];
    out[i] = toInt16(sum / (end - start));
  }
  return out;
}

export function int16ToBase64(i16) {
  const bytes = new Uint8Array(i16.buffer, i16.byteOffset, i16.byteLength);
  let binary = '';
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}

export function base64ToInt16(b64) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length - (binary.length % 2));
  for (let i = 0; i < bytes.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Int16Array(bytes.buffer);
}
