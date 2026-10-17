# SPDX-License-Identifier: MIT
# See LICENSE, LICENSE.fla and THIRD_PARTY_NOTICES.txt.
"""CyFA decoder blocks and causal language model."""

import math

import jittor as jt
from jittor import nn

from ...layers.cyfa import CyclicFlowAttention, Linear, RMSNorm


class GatedMLP(nn.Module):
    def __init__(self, config):
        mid = config.intermediate_size or 256 * math.ceil(
            int(config.hidden_size * config.hidden_ratio * 2 / 3) / 256
        )
        self.gate_proj = Linear(config.hidden_size, mid, bias=False)
        self.up_proj = Linear(config.hidden_size, mid, bias=False)
        self.down_proj = Linear(mid, config.hidden_size, bias=False)

    def execute(self, x):
        g = self.gate_proj(x)
        u = self.up_proj(x)
        return self.down_proj(
            (g.float32() * g.float32().sigmoid() * u.float32()).cast(x.dtype)
        )


class CyFABlock(nn.Module):
    def __init__(self, config):
        self.attn_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.attn = CyclicFlowAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            expand_k=config.expand_k,
            expand_v=config.expand_v,
            num_slots=config.num_slots,
            use_short_conv=config.use_short_conv,
            conv_size=config.conv_size,
            conv_bias=config.conv_bias,
            norm_eps=config.norm_eps,
            chunk_size=config.chunk_size,
        )
        self.mlp_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.mlp = GatedMLP(config)

    def execute(self, x, cache=None, use_cache=False, attention_mask=None):
        y, cache = self.attn(self.attn_norm(x), cache, use_cache, attention_mask)
        residual = x + y
        return residual + self.mlp(self.mlp_norm(residual)), cache


class CyclicFlowAttentionModel(nn.Module):
    def __init__(self, config):
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [CyFABlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)

    def execute(self, input_ids, cache=None, use_cache=False, attention_mask=None):
        x = self.embeddings(input_ids)
        new = []
        for i, layer in enumerate(self.layers):
            x, s = layer(
                x, None if cache is None else cache[i], use_cache, attention_mask
            )
            new.append(s)
        return self.norm(x), new if use_cache else None


class CyclicFlowAttentionForCausalLM(nn.Module):
    """CyFA causal language model with Torch-checkpoint-compatible parameter names."""

    def __init__(self, config):
        self.config = config
        self.model = CyclicFlowAttentionModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embeddings.weight

    @classmethod
    def from_pretrained(cls, path):
        from ...utils import load_model

        model, _ = load_model(path)
        return model

    def save_pretrained(self, directory, tokenizer=None):
        from ...utils import save_model

        save_model(self, directory, tokenizer)

    def execute(
        self, input_ids, labels=None, cache=None, use_cache=False, attention_mask=None
    ):
        hidden, cache = self.model(input_ids, cache, use_cache, attention_mask)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = nn.cross_entropy_loss(
                logits[:, :-1].float32().reshape((-1, self.config.vocab_size)),
                labels[:, 1:].reshape((-1,)),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": loss, "cache": cache}

    def to_bfloat16(self):
        for name, value in self.state_dict().items():
            if not any(
                s in name for s in ("norm.weight", "readout", "A_log", "dt_bias")
            ):
                value.assign(value.bfloat16())
        return self

    def generate(self, input_ids, max_new_tokens=32, eos_token_id=None):
        self.eval()
        tokens = input_ids
        cache = None
        current = input_ids
        with jt.no_grad():
            for _ in range(max_new_tokens):
                result = self(current, cache=cache, use_cache=True)
                cache = result["cache"]
                current = result["logits"][:, -1:].argmax(-1)[0].int32()
                tokens = jt.concat([tokens, current], dim=1)
                if eos_token_id is not None and bool(
                    (current == eos_token_id).all().item()
                ):
                    break
        return tokens
