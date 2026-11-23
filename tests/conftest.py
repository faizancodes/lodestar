"""Shared lightweight fixtures for hermetic Python unit tests (no model downloads or GPU required)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from interp.engine import Engine


class TinyAttention(nn.Module):
    def __init__(self, hidden: int, num_heads: int) -> None:
        super().__init__()
        self.head_dim = hidden // num_heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(self.v_proj(x))


class TinyMLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyLayer(nn.Module):
    def __init__(self, hidden: int, intermediate: int, num_heads: int) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden)
        self.self_attn = TinyAttention(hidden, num_heads)
        self.post_attention_layernorm = nn.LayerNorm(hidden)
        self.mlp = TinyMLP(hidden, intermediate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class TinyBackbone(nn.Module):
    def __init__(self, vocab: int, hidden: int, intermediate: int, num_layers: int, num_heads: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([TinyLayer(hidden, intermediate, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ) -> SimpleNamespace:
        x = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        hidden_states = [x]
        for layer in self.layers:
            x = layer(x)
            hidden_states.append(x)
        last = self.norm(x)
        return SimpleNamespace(
            last_hidden_state=last,
            hidden_states=tuple(hidden_states) if output_hidden_states else None,
            past_key_values=past_key_values if use_cache else None,
        )


class TinyCausalLM(nn.Module):
    base_model_prefix = "model"

    def __init__(self, vocab: int = 32, hidden: int = 8, intermediate: int = 12, num_layers: int = 3, num_heads: int = 2) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.config = SimpleNamespace(
            model_type="tiny",
            hidden_size=hidden,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            hidden_act="silu",
            max_position_embeddings=256,
            rms_norm_eps=1e-5,
            tie_word_embeddings=False,
            mlp_bias=False,
            attention_bias=False,
        )
        self.model = TinyBackbone(vocab, hidden, intermediate, num_layers, num_heads)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        logits_to_keep: int | None = None,
    ) -> SimpleNamespace:
        out = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )
        hidden = out.last_hidden_state
        if logits_to_keep is not None and logits_to_keep > 0:
            hidden = hidden[:, -logits_to_keep:, :]
        return SimpleNamespace(
            logits=self.lm_head(hidden),
            hidden_states=out.hidden_states,
            past_key_values=out.past_key_values,
        )


class TinyTokenizer:
    """Minimal deterministic tokenizer matching the Hugging Face API surface used by Engine."""

    eos_token_id = 1
    pad_token_id = 0
    unk_token_id = 2
    all_special_ids = frozenset({0, 1, 2, 3, 4})
    chat_template = "tiny"

    def __init__(self, vocab: int = 32) -> None:
        self.vocab = vocab

    def _encode_str(self, text: str) -> list[int]:
        if not text:
            return []
        return [5 + (sum(ord(ch) for ch in word) % (self.vocab - 5)) for word in text.split()]

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_tensors: str | None = None,
        return_offsets_mapping: bool = False,
    ) -> Any:
        ids = self._encode_str(text) or [5]
        if return_tensors == "pt":
            return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))
        if return_offsets_mapping:
            offsets = []
            cursor = 0
            for word in text.split():
                start = text.find(word, cursor)
                offsets.append((start, start + len(word)))
                cursor = start + len(word)
            return {"input_ids": ids, "offset_mapping": offsets}
        return SimpleNamespace(input_ids=ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
        tokenize: bool = True,
    ) -> list[int]:
        ids = [3]
        for msg in messages:
            ids.extend(self._encode_str(msg["content"]))
        if add_generation_prompt:
            ids.extend([4, 4])
        return ids

    def convert_tokens_to_ids(self, tok: str) -> int:
        return self.unk_token_id

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        out = []
        for idx in ids:
            if skip_special_tokens and idx in self.all_special_ids:
                continue
            out.append(f"t{idx}")
        return " ".join(out)


@pytest.fixture
def tiny_engine() -> Engine:
    """Return an Engine pre-populated with a tiny deterministic CPU model and hooks installed."""
    eng = Engine()
    model = TinyCausalLM().eval()
    model.requires_grad_(False)
    tok = TinyTokenizer()

    eng.model = model
    eng.tok = tok
    eng.model_id = "test/tiny-model"
    eng.dev = "cpu"
    eng.n_params = sum(p.numel() for p in model.parameters())
    eng.layers = eng._find_layers(model)
    eng.final_norm = eng._find_norm(model)
    eng.lm_head = model.get_output_embeddings()
    eng.backbone = model.model
    eng.status = "ready"

    for i, layer in enumerate(eng.layers):
        eng._handles.append(layer.mlp.down_proj.register_forward_pre_hook(eng._mlp_hook(i)))
        eng._handles.append(layer.register_forward_hook(eng._resid_hook(i)))

    with torch.no_grad():
        eng.dnorm = torch.stack([layer.mlp.down_proj.weight.float().norm(dim=0) for layer in eng.layers])
    eng.resid_p50 = eng._measure_resid_norms()
    eng.unit = eng.resid_p50[:, None] / eng.dnorm.clamp(min=1e-6)
    eng.end_ids = eng._end_ids()
    yield eng
    eng.unload()
