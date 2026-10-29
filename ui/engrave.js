// The page's ground: an engraved plate. A fragment shader draws hatching the way an engraver cuts it: parallel
// lines that bend with the form, thicken where the tone is darker, and gain a crossing set in the darkest parts.
// The tone is light in the middle of the working area and deepens toward its edges under soft clouds, so text
// never sits on heavy ink. It renders once (and again after a resize) into an image used as a CSS background,
// so it costs nothing while a model runs. Without WebGL the plain paper and grain stay.
const VERT = `attribute vec2 p; void main() { gl_Position = vec4(p, 0.0, 1.0); }`;

const FRAG = `
precision highp float;
uniform vec2 res;      // canvas size, css px
uniform vec4 area;     // the working area (x, y, w, h), css px, y down
uniform float dpr;
uniform vec3 paper, ink;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p), u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}
float fbm(vec2 p) {
  float v = 0.0, a = 0.5;
  for (int i = 0; i < 5; i++) { v += a * noise(p); p = p * 2.03 + 17.0; a *= 0.5; }
  return v;
}

// one set of parallel lines across direction ang, spacing s css px, each w (fraction of s) thick, antialiased.
// Each line breaks where the burin lifts, on its own schedule along its length (gaps 0: unbroken).
float lines(vec2 p, float ang, float s, float w, float warp, float gaps) {
  vec2 n = vec2(cos(ang), sin(ang));
  float d = dot(p, n) / s + warp;
  float dist = abs(fract(d) - 0.5) * s;  // css px from the nearest line's centre
  float half_w = 0.5 * w * s, aa = 0.6 / dpr;
  float line = 1.0 - smoothstep(half_w - aa, half_w + aa, dist);
  float lift = smoothstep(gaps, gaps + 0.12, noise(vec2(dot(p, vec2(-n.y, n.x)) / 46.0, floor(d) * 3.7)));
  return gaps > 0.0 ? line * lift : line;
}

void main() {
  vec2 p = vec2(gl_FragCoord.x, res.y * dpr - gl_FragCoord.y) / dpr;   // css px, y down
  vec2 uv = (p - area.xy) / area.zw;
  // tone: 0 is bare paper; light falls from the upper left of the working area
  float r = length((uv - vec2(0.44, 0.34)) * vec2(1.0, 1.15));
  float cloud = fbm(p / 300.0);
  float tone = clamp(smoothstep(0.42, 1.05, r) * 0.9 + (cloud - 0.45) * 0.5, 0.0, 1.0);

  // the main set bends with the form; the crossing set only enters the deeper shadows
  float warp = fbm(p / 420.0 + 3.0) * 12.0;
  float a = lines(p, radians(-34.0), 4.0, mix(0.06, 0.46, tone), warp, 0.24 - 0.2 * tone);
  float b = lines(p, radians(52.0), 4.6, mix(0.0, 0.4, smoothstep(0.5, 0.95, tone)), warp * 0.55, 0.3);
  float cut = max(a, b * 0.85);

  float grain = (hash(floor(p * dpr)) - 0.5) * 0.035;
  float k = clamp(cut * (0.05 + 0.24 * tone) + grain, 0.0, 1.0);
  gl_FragColor = vec4(mix(paper, ink, k), 1.0);
}`;

const rgb = (css) => {
  const el = document.createElement("i");
  el.style.color = css;
  document.body.appendChild(el);
  const m = getComputedStyle(el).color.match(/[\d.]+/g).map(Number);
  el.remove();
  return m.slice(0, 3).map((v) => v / 255);
};

function render(areaEl) {
  const dpr = Math.min(devicePixelRatio || 1, 2), w = innerWidth, h = innerHeight;
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const gl = canvas.getContext("webgl", { preserveDrawingBuffer: true, antialias: false });
  if (!gl) return null;
  const sh = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  };
  const prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, VERT));
  gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(prog);
  gl.useProgram(prog);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, "p");
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  const r = areaEl.getBoundingClientRect();
  const css = getComputedStyle(document.documentElement);
  gl.uniform2f(gl.getUniformLocation(prog, "res"), w, h);
  gl.uniform4f(gl.getUniformLocation(prog, "area"), r.left, r.top, r.width || w, r.height || h);
  gl.uniform1f(gl.getUniformLocation(prog, "dpr"), dpr);
  gl.uniform3fv(gl.getUniformLocation(prog, "paper"), rgb(css.getPropertyValue("--cream")));
  gl.uniform3fv(gl.getUniformLocation(prog, "ink"), rgb(css.getPropertyValue("--ink")));
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  return canvas;
}

/** Engraves the page behind `areaEl` and keeps it fitted to the window. */
export function engrave(areaEl) {
  let url = null, timer = 0;
  const paint = () => {
    let canvas;
    try { canvas = render(areaEl); } catch (e) { console.warn("engraving unavailable:", e); return; }
    if (!canvas) return;
    canvas.toBlob((blob) => {
      if (!blob) return;
      const next = URL.createObjectURL(blob);
      document.documentElement.style.setProperty("--plate", `url("${next}")`);
      if (url) setTimeout((old) => URL.revokeObjectURL(old), 1000, url);
      url = next;
    });
  };
  paint();
  addEventListener("resize", () => { clearTimeout(timer); timer = setTimeout(paint, 250); });
}
