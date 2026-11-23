export { hash } from "../../web/src/art/math";

export type V3 = [number, number, number];

export function clamp(x: number, a = 0, b = 1): number {
  return Math.max(a, Math.min(b, x));
}

export function ease(x: number): number {
  const c = clamp(x);
  return c * c * (3 - 2 * c);
}

export function dot(a: V3, b: V3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function norm(a: V3): V3 {
  const l = Math.hypot(...a) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
}

export function cross3(a: V3, b: V3): V3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
