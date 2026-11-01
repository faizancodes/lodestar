// What the engine (interp/server.py) sends back, written from the handlers there; they return plain dicts.

/** GET /api/status, and what POST /api/load returns. */
export type Arch = {
  heads: number | null; kv_heads: number | null; vocab: number; tied: boolean; experts: number | null;
  attention: ("full" | "linear")[];
};

export type Status = {
  loaded: boolean;
  status: string; // "idle", "loading…", "ready", or "error: …"
  model?: string;
  device?: string;
  dtype?: string;
  params?: number;
  layers?: number;
  neurons?: number;
  hidden?: number;
  /** the model's shape: attention heads, the vocabulary, whether the unembedding is the embedding, and each layer's
   *  attention ("full" over every earlier token, or "linear", a running state) */
  arch?: Arch;
  progress?: number | null;
  /** while loading: what stage it's at, the model's layer count, how far each layer's weights are in (0..1), and
   * the weights' size on disk (null when they're still to download) */
  load?: { stage: "open" | "weights" | "measure"; layers: number | null; lit: number[] | null; bytes: number | null };
  last?: string | null; // the model to open again at launch
};

/** A process holding memory, for the meter and the fit check. */
export type MemProc = { pid: number; name: string; bytes: number };
/** GET /api/memory */
export type Memory = {
  total: number;
  used: number; // what the Mac has in use, of total
  app: number; // Lodestar's own footprint, the model included
  others: MemProc[];
  swap: number;
  pressure: "normal" | "warn" | "critical";
  swapping: boolean; // pages went out to swap in the last few seconds
};
/** POST /api/memory/fit: whether a model fits beside the heavy processes already running */
export type Fit = { known: boolean; fits: boolean; need?: number; free?: number; total?: number; used?: number; others?: MemProc[] };

type Preset = { id: string; size: string; note: string };
type DriveModel = { path: string; name: string; drive: string; size: number; copied: boolean };
type ModelCopy = { path: string; name: string; size: number };
/** GET /api/models */
export type Models = { presets: Preset[]; cached: string[]; local?: DriveModel[]; copies?: ModelCopy[] };

/** GET/POST /api/models/copy */
export type CopyJob = { src: string | null; name: string | null; done: number; total: number; status: string };

/** GET/POST /api/abl/export and POST /api/steer/export */
export type ExportJob = {
  kind: "abl" | "steer" | null;
  name: string | null;
  dst: string | null;
  status: string; // "idle", "working", "done", or "error: …"
  done: number;
  total: number;
  msg: string;
};

/** A neuron's push: the words it moves the next token toward ("up") or away from ("down"). */
type Lens = { up: string[]; down: string[] };

/** A neuron as the searches and traces list it. */
export type NeuronRow = { layer: number; neuron: number; lens: string[]; score?: number; act?: number };
/** GET /api/search */
export type WordSearch = { results: (NeuronRow & { score: number })[]; partial: boolean; piece: string };

/** GET /api/neuron */
export type NeuronDetail = {
  layer: number;
  neuron: number;
  lens: Lens;
  trace?: { tokens: string[]; n_prompt: number; acts: number[] };
};

/** GET /api/trace/token */
export type TokenDetail = {
  lens: { tok: string; p: number }[][];
  jlens: { tok: string; p: number }[][] | null; // null until the model's J-lens is computed
  neurons: NeuronRow[];
};

/** GET /api/jlens: the loaded model's J-lens, and the job computing it (status "idle", "computing", "done" or
 *  "error: …"). */
export type JLensJob = { ready: boolean; status: string; done: number; total: number; elapsed: number };

/** One line of a recipe: a neuron pushed by `dose`, or a direction pushed along or erased over `layers`. */
export type NeuronItem = { kind?: undefined; layer: number; neuron: number; dose: number; label?: string; lens?: string[]; mode?: string };
export type DirItem = { kind: "dir"; dir: string; name?: string; layers: number[]; dose: number; op: "push" | "erase" };
export type Item = NeuronItem | DirItem;
export type Recipe = { id: string | null; name: string; items: Item[]; model?: string; prompt?: string; created?: number };
/** GET /api/examples: known neurons for the loaded model. */
export type Example = { name: string; note: string; prompt: string; items: NeuronItem[] };

/** What the steering endpoints take: a recipe line as the engine applies it. */
export type Push =
  | { layer: number; neuron: number; dose: number; enabled?: boolean }
  | { kind: "dir"; dir: string; layers: number[]; op: "push" | "erase"; dose: number; enabled: boolean };

/** A row of GET /api/directions. */
export type Direction = {
  id: string;
  name: string;
  kind: string;
  band: number[];
  created: number;
  pos?: unknown;
  neg?: unknown;
  ready: boolean; // the vectors exist for the loaded model
  auc: number | null;
};

export type Directions = { directions: Direction[]; layers: number };

/** GET /api/abl */
export type Abl = {
  model: string | null;
  probes: Probes;
  defaults: Probes;
  current: { layer: number | null; coef: number; active: boolean; clean: boolean } | null;
  ready: boolean;
};

export type Probes = { harmful: string[]; harmless: string[] };

export type ChatMessage = { role: "user" | "assistant"; content: string };
/** GET /api/chats: a saved conversation, as the sidebar lists it */
export type ChatSummary = { id: string; title: string; model: string | null; updated: number; turns: number };

/** One event of a stream. `error`, `cancelled` and `end` never reach a handler (see stream.ts). The streams that
 * carry text send deltas; the planning and analysis streams send richer events, typed where their views use them. */
export type StreamEvent = { type: string; [k: string]: unknown };

// The Architecture page (interp/archview.py)
export type ArchTensor = { name: string; shape: number[]; params: number; rms: number };
export type ArchBlock = { name: string; role: "attention" | "mlp" | "norm" | "other"; type: string; params: number; tensors: ArchTensor[] };
export type ArchLayer = { index: number; attention: "full" | "linear"; heads: number[]; head_dim: number | null; blocks: ArchBlock[]; params: number; resid: number };
type ArchPart = { tensors: ArchTensor[]; params: number };
export type ArchDetail = {
  model: string; type: string; dtype: string; params: number; hidden: number; neurons: number; vocab: number;
  activation: string | null; context: number | null; rope_theta: number | null; norm_eps: number | null;
  heads: number | null; kv_heads: number | null; linear_heads: number | null; linear_k_heads: number | null;
  tied: boolean; experts: number | null;
  embedding: ArchPart; final_norm: ArchPart; unembedding: ArchPart; layers: ArchLayer[];
};
export type LensTok = { tok: string; p: number };
/** Text run through the model as a chat message, then the reply it wrote (from n_prompt on): per layer, per token. */
export type ArchRun = { tokens: string[]; n_prompt: number; special: boolean[]; resid: number[][]; attn: number[][]; mlp: number[][]; heads: number[][][]; lens: LensTok[][][] };
/** What every layer did to choose one token of a reply, sent as the model writes it. */
export type ArchLive = {
  resid: number[]; attn: number[]; mlp: number[]; heads: number[][];
  top: [number, number][][]; peak: number[]; lens: (string | null)[];
};
export type ArchActs = { layers: number; neurons: number; peak: number[]; q: string; top: { layer: number; neuron: number; impact: number; act: number }[] };
export type ArchNeuron = { layer: number; neuron: number; norm: number; lens: { up: string[]; down: string[] }; acts: number[] | null };
