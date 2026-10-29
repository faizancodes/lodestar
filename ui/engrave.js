// Engraving, the way the reference etching is made: every tone is lines. Light passages are sparse parallel
// hairlines, mid-tones a mesh of two families of curved lines crossing, shadows a third set on top, and even the
// blacks are lines packed tight. One fragment shader draws it and is rendered twice per window size:
//   --plate    ink on parchment, the page's ground: almost bare paper in the column where text sits, deepening
//              into hatching out in the margins like an engraved vignette, so reading never competes with it
//   --ink-tex  parchment hairlines on night, the texture of every dark surface (sidebar, buttons, the user's
//              messages, steered headers), so blacks read as engraved rather than flat
// Both become CSS background images, so nothing runs per frame while a model works. Without WebGL the flat
// colours stay.
const VERT = `attribute vec2 p; void main() { gl_Position = vec4(p, 0.0, 1.0); }`;

const FRAG = `
precision highp float;
uniform vec2 res;      // canvas size, css px
uniform vec4 area;     // the working area (x, y, w, h), css px, y down
uniform vec2 col;      // the text column inside it: centre x and half width, css px
uniform float dpr;
uniform vec3 bg, fg;
uniform vec2 tone;     // lightest and darkest tone used
uniform float strength, lit;  // how dark a full line prints; 1 shades by the light falling on the area, 0 evenly

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

// One family of engraved lines: across direction ang, s css px apart, w (fraction of s) thick. warp bends the
// whole family with the form; wave ripples each line along its length, which is what weaves two crossing
// families into the etching's mesh.
float family(vec2 p, float ang, float s, float w, float warp, float wave) {
  vec2 n = vec2(cos(ang), sin(ang));
  float along = dot(p, vec2(-n.y, n.x));
  float d = dot(p, n) / s + warp + wave * sin(along / 21.0 + warp * 0.8);
  float dist = abs(fract(d) - 0.5) * s;  // css px from the nearest line's centre
  float half_w = 0.5 * w * s, aa = 0.55 / dpr;
  return 1.0 - smoothstep(half_w - aa, half_w + aa, dist);
}

void main() {
  vec2 p = vec2(gl_FragCoord.x, res.y * dpr - gl_FragCoord.y) / dpr;   // css px, y down
  vec2 uv = (p - area.xy) / area.zw;
  float cloud = fbm(p / 280.0);
  // the column stays light; the shading gathers outside it, heavier toward the bottom, under soft clouds
  float edge = smoothstep(col.y - 12.0, col.y + 200.0, abs(p.x - col.x));
  float framed = edge * (0.5 + 0.7 * (cloud - 0.3)) + 0.25 * edge * smoothstep(0.5, 1.0, uv.y);
  float shade = clamp(mix(cloud, framed, lit), 0.0, 1.0);
  float t = mix(tone.x, tone.y, shade);

  float w1 = fbm(p / 240.0) * 9.0, w2 = fbm(p / 300.0 + 5.0) * 9.0;
  float a = family(p, radians(-28.0), 3.2, clamp(0.1 + t * 0.55, 0.0, 0.7), w1, 0.1);
  float b = family(p, radians(36.0), 3.5, clamp((t - 0.16) * 0.8, 0.0, 0.62), w2, 0.18);
  float c = family(p, radians(82.0), 4.0, clamp((t - 0.6) * 1.2, 0.0, 0.5), w1 * 0.5, 0.2);
  float cut = max(max(a, b), c);

  float grain = (hash(floor(p * dpr)) - 0.5) * 0.03;
  float ink = strength * mix(1.0, mix(0.4, 1.0, edge), lit);  // fainter still inside the column
  gl_FragColor = vec4(mix(bg, fg, clamp(cut * ink + grain, 0.0, 1.0)), 1.0);
}`;

const rgb = (css) => {
  const el = document.createElement("i");
  el.style.color = css;
  document.body.appendChild(el);
  const m = getComputedStyle(el).color.match(/[\d.]+/g).map(Number);
  el.remove();
  return m.slice(0, 3).map((v) => v / 255);
};

function render(areaEl, look) {
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
  const r = areaEl.getBoundingClientRect(), u = (n) => gl.getUniformLocation(prog, n);
  gl.uniform2f(u("res"), w, h);
  gl.uniform4f(u("area"), r.left, r.top, r.width || w, r.height || h);
  gl.uniform2f(u("col"), r.left + (r.width || w) / 2, Math.min((r.width || w) / 2 - 24, 480));  // .page is 920 wide
  gl.uniform1f(u("dpr"), dpr);
  gl.uniform3fv(u("bg"), look.bg);
  gl.uniform3fv(u("fg"), look.fg);
  gl.uniform2f(u("tone"), ...look.tone);
  gl.uniform1f(u("strength"), look.strength);
  gl.uniform1f(u("lit"), look.lit);
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  return canvas;
}

/** Engraves the page around `areaEl` and the dark surfaces, and keeps both fitted to the window. */
export function engrave(areaEl) {
  const urls = {};
  let timer = 0;
  const paint = () => {
    const css = getComputedStyle(document.documentElement), c = (n) => rgb(css.getPropertyValue(n));
    const looks = {
      "--plate": { bg: c("--cream"), fg: c("--ink"), tone: [0.08, 0.95], strength: 0.36, lit: 1 },
      "--ink-tex": { bg: c("--night"), fg: c("--on-night"), tone: [0.02, 0.22], strength: 0.13, lit: 0 },
    };
    for (const [name, look] of Object.entries(looks)) {
      let canvas;
      try { canvas = render(areaEl, look); } catch (e) { console.warn("engraving unavailable:", e); return; }
      if (!canvas) return;
      canvas.toBlob((blob) => {
        if (!blob) return;
        const next = URL.createObjectURL(blob);
        document.documentElement.style.setProperty(name, `url("${next}")`);
        if (urls[name]) setTimeout((old) => URL.revokeObjectURL(old), 1000, urls[name]);
        urls[name] = next;
      });
    }
  };
  paint();
  addEventListener("resize", () => { clearTimeout(timer); timer = setTimeout(paint, 250); });
}
