import * as K from "./kernels";

// Qwen3-0.6B on WebGPU, with the app's hooks: neuron pushes at the input of each MLP's down_proj, the refusal
// direction taken out of each layer's output, and the residual stream and neurons captured at any token.
export const CFG = { H: 1024, L: 28, NH: 16, KVH: 8, HD: 128, N: 3072, V: 151936, CTX: 2048 };
const { H, L, NH, KVH, N, V, CTX } = CFG;
const QD = NH * 128, KD = KVH * 128;
const CHUNK = V / 4;                 // embedding rows per buffer: 4 buffers keep each under the default binding limit
const CHUNK_BYTES = CHUNK * H * 2;
const TT = 8;                        // tokens per matmul tile (kernels.ts)
const LENS = 8;                      // neurons read through the unembedding at once

export const REPO = "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main";
const WEIGHTS = `${REPO}/model.safetensors`;
const CACHE = "lodestar-demo-v1";

export type Push = { layer: number; neuron: number; dose: number };
export type Load = { done: number; total: number; lit: number[] };
export type Lens = { id: number; p: number }[];
export type Inspect = { lens: Lens[]; neurons: { layer: number; neuron: number; score: number }[] };

type Bytes = Uint8Array<ArrayBuffer>;
type Tensor = { name: string; begin: number; end: number; shape: number[] };
const S = GPUBufferUsage.STORAGE, CPY = GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

export async function gpu(): Promise<GPUDevice | string> {
  if (!("gpu" in navigator)) return "This browser has no WebGPU.";
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) return "No GPU adapter is available to this page.";
  return adapter.requestDevice();
}

/** The safetensors header: where each tensor sits in the file's data, and where the data starts. */
async function header() {
  const n = new DataView(await (await fetch(WEIGHTS, { headers: { Range: "bytes=0-7" } })).arrayBuffer()).getUint32(0, true);
  const json = await (await fetch(WEIGHTS, { headers: { Range: `bytes=8-${7 + n}` } })).json();
  const tensors: Tensor[] = Object.entries(json).filter(([k]) => k !== "__metadata__").map(([name, v]: [string, any]) =>
    ({ name, begin: v.data_offsets[0], end: v.data_offsets[1], shape: v.shape }));
  return { base: 8 + n, tensors };
}

export class Qwen {
  dev: GPUDevice;
  w = new Map<string, GPUBuffer>();
  embed: GPUBuffer[] = [];
  unit = new Float32Array(L * N);    // activation per unit dose, as the app's engine.unit
  resid: Float32Array<ArrayBuffer>;
  ablVec: Float32Array<ArrayBuffer>;
  erase = false;

  private buf: Record<string, GPUBuffer> = {};
  private kc: GPUBuffer[] = [];
  private vc: GPUBuffer[] = [];
  private pipes = new Map<string, GPUComputePipeline>();
  private groups = new Map<string, GPUBindGroup>();

  constructor(dev: GPUDevice, pack: { resid_p50: string; abl: { vec: string; coef: number } }) {
    this.dev = dev;
    this.resid = f32(pack.resid_p50);
    this.ablVec = f32(pack.abl.vec);
  }

  // ---------------------------------------------------------------- loading
  /** Stream the weights from Hugging Face (or this browser's cache of them) straight into GPU buffers. */
  async load(progress: (p: Load) => void, signal?: AbortSignal) {
    const { base, tensors } = await header();
    const used = tensors.filter((t) => t.name !== "lm_head.weight"); // tied to the embedding, which is stored twice
    used.sort((a, b) => a.begin - b.begin);
    const from = used[0].begin, to = used[used.length - 1].end;

    for (const t of used) {
      if (t.name === "model.embed_tokens.weight") {
        for (let c = 0; c < 4; c++) this.embed.push(this.dev.createBuffer({ size: CHUNK_BYTES, usage: S | CPY }));
      } else {
        this.w.set(t.name, this.dev.createBuffer({ size: t.end - t.begin, usage: S | CPY }));
      }
    }

    const layerOf = (name: string) => +(/layers\.(\d+)\./.exec(name)?.[1] ?? -1);
    const total = to - from, lit = new Array(L).fill(0), layerBytes = new Array(L).fill(0);
    for (const t of used) if (layerOf(t.name) >= 0) layerBytes[layerOf(t.name)] += t.end - t.begin;

    const body = await this.body(base + from, total, signal);
    const reader = body.getReader();
    let cur = from, ti = 0, pending: Bytes = new Uint8Array(0);

    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      let data: Bytes = value;
      if (pending.length) { data = new Uint8Array(pending.length + value.length); data.set(pending); data.set(value, pending.length); }
      const take = Math.min(data.length - (data.length % 4), to - cur);
      let off = 0;

      while (off < take) {
        while (used[ti].end <= cur) ti++;
        const t = used[ti], n = Math.min(take - off, t.end - cur), at = cur - t.begin;
        if (t.name === "model.embed_tokens.weight") this.writeEmbed(at, data, off, n);
        else this.dev.queue.writeBuffer(this.w.get(t.name)!, at, data, off, n);
        const l = layerOf(t.name);
        if (l >= 0) lit[l] += n / layerBytes[l];
        cur += n; off += n;
      }

      pending = data.slice(take);
      progress({ done: cur - from, total, lit });
    }

    if (cur < to) throw new Error("The download stopped early.");
    await this.setup();
  }

  /** The bytes from `start` to the end of the file: from this browser's cache if they're there, else downloaded
   * once and cached as they arrive. */
  private async body(start: number, size: number, signal?: AbortSignal): Promise<ReadableStream<Bytes>> {
    const key = `${WEIGHTS}?from=${start}`;
    const cache = await caches.open(CACHE).catch(() => null);
    const hit = await cache?.match(key);
    if (hit?.body) return hit.body as ReadableStream<Bytes>;

    const res = await fetch(WEIGHTS, { headers: { Range: `bytes=${start}-` }, signal });
    if (!res.ok || !res.body) throw new Error(`Couldn't download the weights (${res.status}).`);
    let body = res.body as ReadableStream<Bytes>;
    if (res.status === 200) body = body.pipeThrough(skip(start));      // the server sent the whole file
    if (!cache) return body;

    const [mine, keep] = body.tee();
    cache.put(key, new Response(keep, { headers: { "Content-Length": String(size) } })).catch(() => cache.delete(key));
    return mine;
  }

  private writeEmbed(at: number, data: Bytes, off: number, n: number) {
    while (n > 0) {
      const c = Math.floor(at / CHUNK_BYTES), local = at - c * CHUNK_BYTES, k = Math.min(n, CHUNK_BYTES - local);
      this.dev.queue.writeBuffer(this.embed[c], local, data, off, k);
      at += k; off += k; n -= k;
    }
  }

  // ---------------------------------------------------------------- buffers and pipelines
  private async setup() {
    const mk = (name: string, floats: number, usage = S | CPY) => (this.buf[name] = this.dev.createBuffer({ size: floats * 4, usage }));
    mk("ids", CTX); mk("x", CTX * H); mk("n", CTX * H); mk("q", CTX * QD); mk("k", CTX * KD); mk("v", CTX * KD);
    mk("att", CTX * QD); mk("g", CTX * N); mk("u", CTX * N); mk("a", CTX * N);
    mk("add", L * N); mk("vec", H); mk("hn", H); mk("logits", V);
    mk("cap", L * H); mk("acts", L * N); mk("lensN", L * H); mk("lensLogits", L * V);
    mk("cols", LENS * H); mk("colsN", LENS * H); mk("colLogits", LENS * V); mk("dnorm", L * N);
    for (const s of ["st", "head", "lensSt", "colSt"]) this.buf[s] = this.dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.buf.abl = this.dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    for (let l = 0; l < L; l++) {
      this.kc.push(this.dev.createBuffer({ size: CTX * KD * 2, usage: S }));
      this.vc.push(this.dev.createBuffer({ size: CTX * KD * 2, usage: S }));
    }
    this.dev.queue.writeBuffer(this.buf.vec, 0, this.ablVec);
    this.dev.queue.writeBuffer(this.buf.abl, 0, new Float32Array([1, 0, 0, 0]));
    this.step("lensSt", L, 0, 0);
    await this.measure();
  }

  private pipe(kernel: keyof typeof K, constants: Record<string, number>) {
    const key = kernel + JSON.stringify(constants);
    let p = this.pipes.get(key);
    if (!p) {
      const module = this.dev.createShaderModule({ code: K[kernel] });
      p = this.dev.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main", constants } });
      this.pipes.set(key, p);
    }
    return p;
  }

  private group(key: string, p: GPUComputePipeline, entries: (GPUBuffer | GPUBufferBinding)[]) {
    let g = this.groups.get(key);
    if (!g) {
      g = this.dev.createBindGroup({ layout: p.getBindGroupLayout(0), entries: entries.map((r, i) => ({ binding: i, resource: "buffer" in r ? r : { buffer: r } })) });
      this.groups.set(key, g);
    }
    return g;
  }

  private step(name: string, T: number, pos0: number, row0: number) {
    this.dev.queue.writeBuffer(this.buf[name], 0, new Uint32Array([T, pos0, row0, 0]));
  }

  private run(pass: GPUComputePassEncoder, p: GPUComputePipeline, g: GPUBindGroup, x: number, y = 1) {
    pass.setPipeline(p); pass.setBindGroup(0, g); pass.dispatchWorkgroups(x, y);
  }

  private mm(pass: GPUComputePassEncoder, st: string, x: string, w: string, y: string, IN: number, OUT: number, T: number, ACC = false) {
    const p = this.pipe("MATMUL", { IN, YS: OUT, ACC: +ACC });
    this.run(pass, p, this.group(`mm ${st} ${x} ${w} ${y}`, p, [this.buf[st], this.buf[x], this.w.get(w)!, this.buf[y]]), OUT, Math.ceil(T / TT));
  }

  private norm(pass: GPUComputePassEncoder, st: string, x: string, w: GPUBuffer, wname: string, out: string, T: number) {
    const p = this.pipe("RMSNORM", { D: H });
    this.run(pass, p, this.group(`rms ${st} ${x} ${wname} ${out}`, p, [this.buf[st], this.buf[x], w, this.buf[out]]), T);
  }

  /** The unembedding of T normalized rows of `x` into `y` ([T, V]), a quarter of the vocabulary at a time. */
  private head(pass: GPUComputePassEncoder, st: string, x: string, y: string, T: number) {
    for (let c = 0; c < 4; c++) {
      const p = this.pipe("MATMUL", { IN: H, YS: V, ROW0: c * CHUNK });
      this.run(pass, p, this.group(`head ${st} ${x} ${y} ${c}`, p, [this.buf[st], this.buf[x], this.embed[c], this.buf[y]]), CHUNK, Math.ceil(T / TT));
    }
  }

  /** Every layer's down_proj column norms, read back to size a dose as the app does. */
  private async measure() {
    const enc = this.dev.createCommandEncoder(), pass = enc.beginComputePass();
    const p = this.pipe("COLNORM", { D: H, N });
    for (let l = 0; l < L; l++) {
      const g = this.dev.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.w.get(`model.layers.${l}.mlp.down_proj.weight`)! } },
        { binding: 1, resource: { buffer: this.buf.dnorm, offset: l * N * 4, size: N * 4 } }] });
      this.run(pass, p, g, N / 64);
    }
    pass.end();
    this.dev.queue.submit([enc.finish()]);
    const dnorm = await this.read(this.buf.dnorm, L * N);
    for (let l = 0; l < L; l++) for (let n = 0; n < N; n++) this.unit[l * N + n] = this.resid[l] / Math.max(dnorm[l * N + n], 1e-6);
  }

  // ---------------------------------------------------------------- hooks
  setPushes(pushes: Push[]) {
    const add = new Float32Array(L * N);
    for (const p of pushes) add[p.layer * N + p.neuron] += p.dose * this.unit[p.layer * N + p.neuron];
    this.dev.queue.writeBuffer(this.buf.add, 0, add);
  }

  // ---------------------------------------------------------------- the forward pass
  /** The layers over T tokens starting at position pos0, from the ids already in the ids buffer. With `at`, the
   * residual stream after each layer and the neurons at that row are copied out. */
  private layers(enc: GPUCommandEncoder, T: number, at?: number) {
    let pass = enc.beginComputePass();
    const pe = this.pipe("EMBED", { H, CHUNK });
    this.run(pass, pe, this.group("embed", pe, [this.buf.st, this.buf.ids, ...this.embed, this.buf.x]), H / 256, T);

    for (let l = 0; l < L; l++) {
      const w = (s: string) => `model.layers.${l}.${s}.weight`;
      this.norm(pass, "st", "x", this.w.get(w("input_layernorm"))!, w("input_layernorm"), "n", T);
      this.mm(pass, "st", "n", w("self_attn.q_proj"), "q", H, QD, T);
      this.mm(pass, "st", "n", w("self_attn.k_proj"), "k", H, KD, T);
      this.mm(pass, "st", "n", w("self_attn.v_proj"), "v", H, KD, T);

      const pq = this.pipe("QKROPE", { NH }), pk = this.pipe("QKROPE", { NH: KVH });
      this.run(pass, pq, this.group(`rq ${l}`, pq, [this.buf.st, this.buf.q, this.w.get(w("self_attn.q_norm"))!]), NH, T);
      this.run(pass, pk, this.group(`rk ${l}`, pk, [this.buf.st, this.buf.k, this.w.get(w("self_attn.k_norm"))!]), KVH, T);
      const ps = this.pipe("KVSTORE", { KVH });
      this.run(pass, ps, this.group(`kv ${l}`, ps, [this.buf.st, this.buf.k, this.buf.v, this.kc[l], this.vc[l]]), KVH, T);
      const pa = this.pipe("ATTN", { NH, KVH, CTX });
      this.run(pass, pa, this.group(`at ${l}`, pa, [this.buf.st, this.buf.q, this.kc[l], this.vc[l], this.buf.att]), NH, T);
      this.mm(pass, "st", "att", w("self_attn.o_proj"), "x", QD, H, T, true);

      this.norm(pass, "st", "x", this.w.get(w("post_attention_layernorm"))!, w("post_attention_layernorm"), "n", T);
      this.mm(pass, "st", "n", w("mlp.gate_proj"), "g", H, N, T);
      this.mm(pass, "st", "n", w("mlp.up_proj"), "u", H, N, T);
      const pw = this.pipe("SWIGLU", { N });
      this.run(pass, pw, this.group(`sw ${l}`, pw, [this.buf.st, this.buf.g, this.buf.u, { buffer: this.buf.add, offset: l * N * 4, size: N * 4 }, this.buf.a]), N / 256, T);
      this.mm(pass, "st", "a", w("mlp.down_proj"), "x", N, H, T, true);

      if (this.erase) {
        const pr = this.pipe("ERASE", { D: H });
        this.run(pass, pr, this.group("erase", pr, [this.buf.st, this.buf.x, this.buf.vec, this.buf.abl]), T);
      }

      if (at !== undefined) {
        pass.end();
        enc.copyBufferToBuffer(this.buf.x, at * H * 4, this.buf.cap, l * H * 4, H * 4);
        enc.copyBufferToBuffer(this.buf.a, at * N * 4, this.buf.acts, l * N * 4, N * 4);
        pass = enc.beginComputePass();
      }
    }
    return pass;
  }

  /** Run T new tokens at pos0 and return the next token's logits. */
  private async next(ids: number[], pos0: number) {
    const T = ids.length;
    this.dev.queue.writeBuffer(this.buf.ids, 0, new Uint32Array(ids));
    this.step("st", T, pos0, 0);
    this.step("head", 1, 0, T - 1);

    const enc = this.dev.createCommandEncoder();
    const pass = this.layers(enc, T);
    this.norm(pass, "head", "x", this.w.get("model.norm.weight")!, "model.norm.weight", "hn", 1);
    this.head(pass, "head", "hn", "logits", 1);
    pass.end();
    this.dev.queue.submit([enc.finish()]);
    return this.read(this.buf.logits, V);
  }

  /** Each next token in turn, until an end token, `max` tokens, or the signal. */
  async *generate(ids: number[], opts: { max: number; temp: number; stop: Set<number>; signal?: AbortSignal }) {
    if (ids.length >= CTX) throw new Error("This conversation is longer than the demo can hold. Start a new chat.");
    let logits = await this.next(ids, 0), pos = ids.length;

    for (let i = 0; i < opts.max && pos < CTX; i++) {
      if (opts.signal?.aborted) return;
      const tok = sample(logits, opts.temp);
      if (opts.stop.has(tok)) return;
      yield tok;
      logits = await this.next([tok], pos++);
    }
  }

  /** What the model holds at row `at` of `ids`: the logit lens after each layer, and the neurons firing hardest, in
   * dose units (how far each moves the residual stream, against a typical residual's size). */
  async inspect(ids: number[], at: number): Promise<Inspect> {
    const T = at + 1;
    this.dev.queue.writeBuffer(this.buf.ids, 0, new Uint32Array(ids.slice(0, T)));
    this.step("st", T, 0, 0);

    const enc = this.dev.createCommandEncoder();
    const pass = this.layers(enc, T, at);
    this.norm(pass, "lensSt", "cap", this.w.get("model.norm.weight")!, "model.norm.weight", "lensN", L);
    this.head(pass, "lensSt", "lensN", "lensLogits", L);
    pass.end();
    this.dev.queue.submit([enc.finish()]);

    const [logits, acts] = await Promise.all([this.read(this.buf.lensLogits, L * V), this.read(this.buf.acts, L * N)]);
    const lens = Array.from({ length: L }, (_, l) => topP(logits.subarray(l * V, (l + 1) * V), 3));

    const all: { layer: number; neuron: number; score: number }[] = [];
    for (let l = 0; l < L; l++) for (let n = 0; n < N; n++) {
      const score = acts[l * N + n] / this.unit[l * N + n];
      if (Math.abs(score) > 0.05) all.push({ layer: l, neuron: n, score });
    }
    all.sort((a, b) => Math.abs(b.score) - Math.abs(a.score));
    return { lens, neurons: all.slice(0, 8) };
  }

  /** The tokens each neuron's push moves the model toward: its down_proj column (negated for a negative push),
   * through the final norm and the unembedding. */
  async neuronLens(items: { layer: number; neuron: number; sign: number }[]) {
    items = items.slice(0, LENS);
    if (!items.length) return [];
    const enc = this.dev.createCommandEncoder();
    let pass = enc.beginComputePass();
    const p = this.pipe("COLUMN", { D: H, N });

    items.forEach((it, r) => {
      const u = this.dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.dev.queue.writeBuffer(u, 0, new Uint32Array([it.neuron, r, 0, 0]));
      this.dev.queue.writeBuffer(u, 8, new Float32Array([it.sign]));
      const g = this.dev.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.w.get(`model.layers.${it.layer}.mlp.down_proj.weight`)! } },
        { binding: 1, resource: { buffer: u } }, { binding: 2, resource: { buffer: this.buf.cols } }] });
      this.run(pass, p, g, H / 64);
    });

    this.step("colSt", items.length, 0, 0);
    this.norm(pass, "colSt", "cols", this.w.get("model.norm.weight")!, "model.norm.weight", "colsN", items.length);
    this.head(pass, "colSt", "colsN", "colLogits", items.length);
    pass.end();
    this.dev.queue.submit([enc.finish()]);
    const logits = await this.read(this.buf.colLogits, items.length * V);
    return items.map((_, r) => topP(logits.subarray(r * V, (r + 1) * V), 6).map((x) => x.id));
  }

  private async read(src: GPUBuffer, floats: number) {
    const st = this.dev.createBuffer({ size: floats * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.dev.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, st, 0, floats * 4);
    this.dev.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(st.getMappedRange().slice(0));
    st.destroy();
    return out;
  }
}

// ---------------------------------------------------------------- helpers
function f32(b64: string) {
  const s = atob(b64), b = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) b[i] = s.charCodeAt(i);
  return new Float32Array(b.buffer);
}

/** Drop the first `n` bytes of a stream. */
function skip(n: number) {
  return new TransformStream<Bytes, Bytes>({
    transform(chunk, c) {
      if (n >= chunk.length) { n -= chunk.length; return; }
      c.enqueue(n ? chunk.subarray(n) : chunk);
      n = 0;
    },
  });
}

/** The k likeliest tokens with their probabilities. */
export function topP(logits: Float32Array, k: number): Lens {
  const best: { id: number; l: number }[] = [];
  let mx = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    const l = logits[i];
    if (l > mx) mx = l;
    if (best.length < k || l > best[best.length - 1].l) {
      best.push({ id: i, l });
      best.sort((a, b) => b.l - a.l);
      if (best.length > k) best.pop();
    }
  }
  let z = 0;
  for (let i = 0; i < logits.length; i++) z += Math.exp(logits[i] - mx);
  return best.map((b) => ({ id: b.id, p: Math.exp(b.l - mx) / z }));
}

/** Qwen3's recommended sampling without thinking (top-k 20, top-p 0.8) at `temp`; greedy at 0. */
function sample(logits: Float32Array, temp: number) {
  const top = topP(logits, 20);
  if (temp <= 0) return top[0].id;
  const w = top.map((t) => Math.pow(t.p, 1 / temp)), z = w.reduce((a, b) => a + b, 0);
  let cum = 0;
  const keep: { id: number; w: number }[] = [];
  for (let i = 0; i < top.length; i++) { keep.push({ id: top[i].id, w: w[i] / z }); cum += w[i] / z; if (cum >= 0.8) break; }
  const zk = keep.reduce((a, b) => a + b.w, 0);
  let r = Math.random() * zk;
  for (const k of keep) if ((r -= k.w) <= 0) return k.id;
  return keep[0].id;
}
