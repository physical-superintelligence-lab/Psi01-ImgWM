"""Text-tower (understanding, "und") KV cache across the steps of one sampling request.

With ``two_way`` attention the text tokens attend only to text tokens, and the timestep is added
only to the noisy vision tokens, so every layer's text keys and values are the same on every
sampling step. The first step runs the full network and stores them; every later step runs the
layers gen-only, attending over the stored text keys and values instead of recomputing the text
tower, which holds most of the network's parameters.

Sampling only, in one process: ``OmniMoTModel.generate_samples_from_batch(und_kv_cache=True)``
builds a fresh cache per request; training never does.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cosmos_framework.data.vfm.sequence_packing.runtime import SequencePack, from_mode_splits, get_full_only_seq
from cosmos_framework.model.attention import attention
from cosmos_framework.model.vfm.utils.memory import KVToStore, MemoryState, MemoryValue


@dataclass
class UndKVMemoryValue(MemoryValue):
    """One layer's cached text keys and values (after RoPE); ``None`` before the first step is done."""

    und_k: torch.Tensor | None  # [N_und, num_kv_heads, head_dim]
    und_v: torch.Tensor | None  # [N_und, num_kv_heads, head_dim]


class UndKVCacheState(MemoryState):
    """The cache of one request: written by the first forward, read by every later one.

    The network calls ``init`` once per forward and ``read_for_layer`` / ``write_for_layer`` around
    every layer. The cache counts as full from the second forward on, once every layer has written.
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self._und_k: list[torch.Tensor | None] = [None] * num_layers
        self._und_v: list[torch.Tensor | None] = [None] * num_layers
        self._populated = False

    def init(self, hidden_states: dict, device: torch.device) -> None:
        del hidden_states, device
        # Decided once per forward, so a forward never switches to gen-only halfway through.
        self._populated = self._und_k[-1] is not None

    def is_gen_only(self) -> bool:
        return self._populated

    def read_for_layer(self, layer_idx: int) -> UndKVMemoryValue:
        if not self._populated:
            return UndKVMemoryValue(und_k=None, und_v=None)
        und_k = self._und_k[layer_idx]
        und_v = self._und_v[layer_idx]
        assert und_k is not None and und_v is not None, f"layer {layer_idx} read before its first-step write"
        return UndKVMemoryValue(und_k=und_k, und_v=und_v)

    def write_for_layer(self, layer_idx: int, kv_to_store: KVToStore) -> None:
        if self._populated:  # later steps still hand over their (empty) text lane
            return
        _gen_k, _gen_v, und_k, und_v = kv_to_store
        # A copy, not a view: under CUDA graphs the next replay overwrites the graph's output memory.
        self._und_k[layer_idx] = und_k.detach().squeeze(0).clone()
        self._und_v[layer_idx] = und_v.detach().squeeze(0).clone()


def cached_two_way_attention(
    packed_query_states: SequencePack,
    packed_key_states: SequencePack,
    packed_value_states: SequencePack,
    memory_value: UndKVMemoryValue,
) -> SequencePack:
    """``two_way_attention`` for a gen-only pack, with the text (und) keys and values from the cache.

    The gen queries attend over the same [text ++ gen] keys and values, with the same kernel and
    offsets, as in ``two_way_attention``; only the text rows come from the first step's cache. The
    text queries are absent: nothing reads the text outputs while sampling.
    """
    und_k, und_v = memory_value.und_k, memory_value.und_v
    assert und_k is not None and und_v is not None

    num_und = int(packed_query_states["_num_causal_tokens"])
    num_gen = int(packed_query_states["_num_full_tokens"])
    assert und_k.shape[0] == num_und, (
        f"cached und K has {und_k.shape[0]} tokens but the pack expects {num_und}; "
        "the pack layout changed between sampling steps"
    )

    full_q, full_q_offsets = get_full_only_seq(packed_query_states)
    k_gen = get_full_only_seq(packed_key_states)[0][:num_gen]  # [N_gen,kv_heads,head_dim]
    v_gen = get_full_only_seq(packed_value_states)[0][:num_gen]  # [N_gen,kv_heads,head_dim]

    causal_indices = packed_query_states["_causal_indices"]
    full_indices = packed_query_states["_full_indices"]
    all_k = k_gen.new_zeros((num_und + num_gen, *k_gen.shape[1:]))  # [N_all,kv_heads,head_dim]
    all_v = v_gen.new_zeros((num_und + num_gen, *v_gen.shape[1:]))  # [N_all,kv_heads,head_dim]
    all_k[causal_indices] = und_k
    all_k[full_indices] = k_gen
    all_v[causal_indices] = und_v
    all_v[full_indices] = v_gen

    full_res = attention(
        full_q.unsqueeze(0),  # [1,N_full,heads,head_dim]
        all_k.unsqueeze(0),  # [1,N_all,kv_heads,head_dim]
        all_v.unsqueeze(0),  # [1,N_all,kv_heads,head_dim]
        cumulative_seqlen_Q=full_q_offsets,
        cumulative_seqlen_KV=packed_query_states["sample_offsets"],
        max_seqlen_Q=packed_query_states["max_full_len"],
        max_seqlen_KV=packed_query_states["max_sample_len"],
    )  # [1,N_full,heads,head_dim]

    # [1,N_full,heads,head_dim] -> [N_full,heads,head_dim] -> [N_full,heads*head_dim]
    full_out = full_res.squeeze(0).flatten(-2, -1)  # type: ignore  # [N_full,heads*head_dim]
    causal_out = full_out.new_zeros((0, full_out.shape[-1]))  # gen-only pack: no und queries
    return from_mode_splits(causal_out, full_out, packed_query_states)
