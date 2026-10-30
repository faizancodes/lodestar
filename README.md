# Lodestar

A local Mac app for mechanistic interpretability on local LLMs. Run a prompt and watch which neurons and layers
light up. Push single neurons up or down, stack them into recipes, or describe a behaviour and let the app search for
the neurons that produce it.

Inspired by [@treetowntree's Qwen neuron zoo](https://x.com/treetowntree/status/2114573600976183723) (single neurons
and pairs that give Qwen a one-track mind, up to a Cookie Monster), [@Sauers_'s gnome neurons](https://x.com/Sauers_/status/2114599600078344380)
(three neurons across layers 3, 4 and 8 of Qwen3-8B), and [lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer),
whose hook placement and dose units this app reuses.

## Run it

    uv sync
    ./scripts/build_app.sh          # builds the UI and makes dist/Lodestar.app; drag it to /Applications or the Dock
    open "dist/Lodestar.app"

or without the bundle:

    uv run python -m interp.app            # native window
    uv run python -m interp.app --server   # headless, http://127.0.0.1:8792

The UI is the React build in `web/dist` (`pnpm install && pnpm build` in `web/`; `build_app.sh` does this).

Pick a model in the sidebar (any Hugging Face id for a Llama-style decoder: Qwen, Llama, Mistral, Gemma, SmolLM).
The first load downloads the weights. Models under 2.5B parameters run in float32 on the Apple GPU (MPS) for precise
gradients; bigger ones in bfloat16. Logs go to `~/Library/Logs/Lodestar.log`, recipes to
`~/Library/Application Support/Auto Interp/recipes.json`.

## What a neuron is here

One unit at the input of a layer's MLP `down_proj`, where `silu(gate) * up` lives. Reading, adding and multiplying
there are exact. An `add` dose is measured in the layer's median residual norm, divided by the neuron's down-column
norm, so dose 1.0 moves the residual stream about one typical residual's length on any layer of any model. Doses of
0.5–0.8 usually show the direction while keeping text coherent; 1.5+ tends to loop.

## Views

- **Trace**: generates a reply and captures every neuron at every token in one forward pass. The heatmap shows MLP
  and attention write norms per layer per token (each layer on its own scale) or logit-lens confidence. Click a token
  for the logit lens at every layer ("what each layer would say next") and the neurons pushing hardest there.
  Impact = |activation| × down-column norm ÷ layer residual median, so neurons compare across layers.
- **Neuron**: logit lens of the down column (tokens promoted / suppressed), activations over the last trace or your
  own text, and three-way steering: pushed, untouched, reversed.
- **Steer**: stack neurons with per-neuron doses and an overall strength; generate beside the untouched and reversed
  model, or trace the steered run.
- **Tune**: the automatic search, below.
- **Recipes**: saved neuron sets, tagged with the model they belong to.
- **Shed its refusals** (abliteration): removes the model's refusal behaviour with the method from
  [Arditi et al. 2024](https://arxiv.org/abs/2406.11717) — refusal is one direction the model represents in the
  residual stream. The analysis runs ~30 refusing and ~30 everyday probe prompts, reads the last-token residual
  state at every layer, takes the harmful−harmless difference-in-means per layer, picks the layer where two
  independent halves of the harmful set agree best (|cos| ≥ ~0.9 is a clean hit), and then:
  - **live** — a forward hook projects the direction out of every layer's residual stream at every token. Reversible
    with one click; Try-it shows Normal beside Ablated, and a check panel scores refusal rates and the fluency cost
    on ordinary text. The strength setting scales how much of the projection is removed (1.0 = fully).
  - **baked** — a fresh copy of the weights is loaded on the CPU and every matrix that writes to the residual
    stream is orthogonalized against the direction (attention `o_proj` and MLP `down_proj` of layers 1…L, the word
    embedding, and the unembedding, which also cleans up the small component the final norm can return to the
    logits). Saved under `~/Library/Application Support/Auto Interp/models/<model>-abliterated/` and picked up by
    the model list like any other local copy.
  Directions, probe edits and the on/off state are saved per model and restored when a model loads.

## How Tune finds neurons

1. **Targets.** Each probe prompt is answered twice: plainly (the default) and with your behaviour as a system
   prompt (the target). Edit any target to say exactly what you want; the last few prompts are held out.
2. **Attribution.** One backward pass per training reply gives every neuron's first-order steering effect on
   `J = logp(target | plain prompt) − contrast · logp(default | plain prompt)`. Steering adds the same value at every
   position, so this gradient is exactly dJ/d(dose).
3. **Attribution patching.** The activation shift the system prompt causes on the same target text, times that
   gradient: which neurons carry the behaviour when the model is told to have it (the
   [tawny-owl](https://huggingface.co/Sauers/Qwen3-8B-tawny-owl-128-neurons) causal search, linearised). Half the
   candidates come from here, half from raw attribution, capped per layer.
4. **Verification.** Every candidate is actually pushed at several doses and J is measured teacher-forced, along
   with the extra loss on a neutral passage. First-order estimates often fail at real doses, so this step decides.
5. **Combination.** Verified neurons are stacked greedily, each kept only if it adds to J, then the recipe's overall
   strength is tuned.
6. **Proof.** The recipe runs on the held-out prompts with no system prompt, beside the untouched model.

What to expect: small models spread most personas over many neurons, so a 0.6B recipe shifts tone and likelihood more
than it produces a full character. The neuron-zoo results used a 4B model. Raise `max neurons`, loosen the fluency
budget, or load Qwen3-1.7B/4B for stronger single-neuron levers.

## Layout

    interp/engine.py     model, hooks, generation, tracing, lens, scoring, gradient attribution
    interp/discover.py   the Tune search
    interp/abl.py        abliteration: probe lists, direction analysis, weight baking
    interp/server.py     local API (FastAPI, server-sent events)
    interp/app.py        native window (pywebview) around the server
    web/                 the interface (React + TypeScript, Vite)
    e2e/                 the parity suite: recorded API sessions, mock replay, golden screenshots
    scripts/             app bundle and icon builders
