<div align="center">

# CyFA: Linear Sequence Modeling with Relative-Time-Partitioned Memory

**[Yixiao Chen](https://chyxx.github.io/) · Shuojin Yang · Shi-Min Hu**<br>
Tsinghua University

[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b)](https://arxiv.org/abs/2609.36259) [![Models](https://img.shields.io/badge/Hugging_Face-Checkpoints-ffd21e)](https://huggingface.co/collections/cyxxxxxxxxxx/cyclicflowattention-6abb56dc723179dddead18b0) [![Zhihu](https://img.shields.io/badge/Zhihu-Intro-blue?logo=zhihu)](https://zhuanlan.zhihu.com/p/2090817772250984628)

</div>

Official Jittor implementation of **Cyclic Flow Attention (CyFA)**. CyFA organizes
a fixed-size recurrent memory by relative time, combining a learned clock and
cyclic transport with content-based softmax retrieval. An exact change to
absolute-clock coordinates yields two scalar-decay recurrences, enabling
linear-time chunkwise training and constant-memory autoregressive decoding.

The architecture and checkpoint format follow the
[PyTorch implementation](https://github.com/Chyxx/CyclicFlowAttention).

<p align="center">
  <img src="assets/cyfa_overview.png" width="100%" alt="Cyclic transport and relative-time-partitioned memory in CyFA">
</p>
<p align="center"><em>
Figure 1. CyFA advances stored associations through relative-time slots before
each new write. A learned clock controls the spacing between writes, while the
same transport keeps key and value memories aligned.
</em></p>

## Method

### Relative-time recurrence

Each head maintains aligned key and value memories. A learned clock increment
$\delta_t\in(0,1)$ transports both memories before the current pair is written
to the age-zero slot:

$$
\begin{aligned}
\mathbf K_t
&=\alpha_t\mathbf P(\delta_t)\mathbf K_{t-1}
+\beta_t\boldsymbol e_0\boldsymbol k_t^\top,\\
\mathbf V_t
&=\alpha_t\mathbf P(\delta_t)\mathbf V_{t-1}
+\beta_t\boldsymbol e_0\boldsymbol v_t^\top.
\end{aligned}
$$

Here $\mathbf P(\delta_t)$ is a fractional cyclic shift, $\alpha_t$ is a scalar
forget gate, and $\beta_t$ controls write strength. A learned matrix $\mathbf R$
combines relative-time slots during content-based retrieval:

$$
\boldsymbol o_t=(\mathbf R\mathbf V_t)^\top
\mathrm{softmax}\left(\mathbf R\mathbf K_t\boldsymbol q_t\right).
$$

### Absolute-clock recurrence

Let $\lambda_t=\lambda_{t-1}+\delta_t$ be the cumulative clock,
$\boldsymbol b=\boldsymbol\Phi^\top\boldsymbol e_0$, and $\mathcal U(\lambda)$ the
block-diagonal Fourier rotation associated with the cyclic shift. Moving the
state into absolute-clock coordinates removes the dense transport from the
recurrent transition exactly:

$$
\begin{aligned}
\overline{\mathbf K}_t
&=\alpha_t\overline{\mathbf K}_{t-1}
+\beta_t\mathcal U(-\lambda_t)\boldsymbol b\boldsymbol k_t^\top,\\
\overline{\mathbf V}_t
&=\alpha_t\overline{\mathbf V}_{t-1}
+\beta_t\mathcal U(-\lambda_t)\boldsymbol b\boldsymbol v_t^\top.
\end{aligned}
$$

The learned clock now controls only the temporal write vector; both stored
states advance through the scalar decay $\alpha_t$.

### Two-pass computation

The equivalent implementation consists of two scalar-gated linear-attention
passes connected by a token-wise softmax readout:

$$
\begin{aligned}
\{\boldsymbol o'_t\}_{t=1}^{T}
&=\mathrm{ScalarGatedLA}\left(
\{\boldsymbol q_t,\boldsymbol k_t,
\beta_t\mathcal U(-\lambda_t)\boldsymbol b,\alpha_t\}_{t=1}^{T}
\right),\\
\boldsymbol o''_t
&=\mathcal U(-\lambda_t)\boldsymbol\Phi^\top\mathbf R^\top
\mathrm{softmax}\left(
\mathbf R\boldsymbol\Phi\mathcal U(\lambda_t)\boldsymbol o'_t
\right),\\
\{\boldsymbol o_t\}_{t=1}^{T}
&=\mathrm{ScalarGatedLA}\left(
\{\boldsymbol o''_t,
\beta_t\mathcal U(-\lambda_t)\boldsymbol b,
\boldsymbol v_t,\alpha_t\}_{t=1}^{T}
\right).
\end{aligned}
$$

The first pass queries the key memory, the middle transformation produces
relative-time slot weights, and the second pass retrieves from the value
memory. Both recurrent passes admit parallel chunkwise training and fused
constant-memory decoding. The released models use 127 active slots, stored
with `num_slots=128` rows per head.

## Checkpoints

Released PyTorch checkpoints load directly in Jittor.

| Model | Config | Training tokens | Checkpoint |
| --- | --- | ---: | --- |
| CyFA-400M | [400M](configs/cyclic_flow_attention_400m.json) | 15B | [Hugging Face](https://huggingface.co/cyxxxxxxxxxx/cyfa-400M-15B) |
| CyFA-800M | [800M](configs/cyclic_flow_attention_800m.json) | 30B | [Hugging Face](https://huggingface.co/cyxxxxxxxxxx/cyfa-800M-30B) |
| CyFA-1.4B | [1.4B](configs/cyclic_flow_attention_1_4b.json) | 100B | [Hugging Face](https://huggingface.co/cyxxxxxxxxxx/cyfa-1.4B-100B) |

## Installation

Requirements: **Linux**, **Python 3.10–3.12**, **Jittor 1.3.11.0**, **CUDA 12.x with cuBLAS**, `g++`, and an **NVIDIA GPU with BF16 support (Ampere or newer)**.

Install in a separate environment from the PyTorch version.

```bash
git clone https://github.com/Chyxx/CyFA-Jittor.git
cd CyFA-Jittor
pip install -e '.[train,eval]'
```

## Quick Start

### Generate from a checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 python -m cyclic_flow_attention.generation \
  --model-path cyxxxxxxxxxx/cyfa-400M-15B \
  --prompt "The library opens at" --max-new-tokens 64
```

### As a layer

```python
import jittor as jt
from cyclic_flow_attention import CyclicFlowAttention
from cyclic_flow_attention.utils import setup_cuda

setup_cuda()
layer = CyclicFlowAttention(
    hidden_size=1024,
    num_heads=4,
    head_dim=256,
    num_slots=128,
).bfloat16()
x = jt.randn((1, 2048, 1024)).bfloat16()
y, cache = layer(x)
assert y.shape == x.shape
```

## Data Preparation

```bash
python -m training.prepare_data \
  --dataset cerebras/SlimPajama-627B --split train --text-column text \
  --tokenizer fla-hub/gla-1.3B-100B --output-dir data/slimpajama
```

## Training

Train CyFA-400M on eight GPUs with a global batch of 256 sequences. Multi-GPU training requires MPI and NCCL development headers and libraries.

```bash
export nccl_include_path=/path/to/nccl/include
export nccl_lib_path=/path/to/nccl/lib
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun -np 8 python -m training.train \
  --config configs/cyclic_flow_attention_400m.json \
  --tokenizer fla-hub/gla-1.3B-100B --data data/slimpajama \
  --output-dir outputs/cyfa-400m --context-length 2048 \
  --batch-size 1 --gradient-accumulation-steps 32 \
  --learning-rate 3e-4 --warmup-steps 1024 --max-steps 30720
```

Training uses BF16 and fixed-length batches (`checkpoint_level=0`). Use `--resume CHECKPOINT` to resume or `--model-path CHECKPOINT` to fine-tune.

## Evaluation

Both evaluations use BF16 and the original task definitions.

```bash
export CHECKPOINT=/path/to/checkpoint
```

### Language modeling and commonsense reasoning

```bash
python -m eval.lm_eval --model-path "$CHECKPOINT" \
  --tasks arc_easy,arc_challenge,hellaswag,lambada_standard,piqa,winogrande,wikitext \
  --max-length 2048 --batch-size 32 --num-fewshot 0 --output-path results/lm.json
```

The commonsense average uses `acc_norm` for ARC-Challenge and HellaSwag, and `acc` for the other four tasks. WikiText perplexity is reported separately.

### Recall-intensive tasks

```bash
python -m eval.recall_eval --model-path "$CHECKPOINT" \
  --tasks based_fda,based_swde,based_squad,based_nq_2048,based_triviaqa,based_drop \
  --context-length 2048 --answer-length 48 --batch-size 32 --output-path results/recall.json
```

Recall uses zero-shot greedy generation, stopping at a newline or after 48 answer tokens.

## License

[MIT](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.txt).

## Acknowledgements

Built upon [CyFA](https://github.com/Chyxx/CyclicFlowAttention), [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention), [Jittor](https://github.com/Jittor/jittor) and [Prefix Linear Attention](https://github.com/HazyResearch/prefix-linear-attention).

## Citation

```bibtex
@misc{chen2026cyfalinearsequencemodeling,
  title={CyFA: Linear Sequence Modeling with Relative-Time-Partitioned Memory},
  author={Yixiao Chen and Shuojin Yang and Shi-Min Hu},
  year={2026},
  eprint={2609.36259},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2609.36259},
}
```
