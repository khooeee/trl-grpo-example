# GRPO + TRL on Modal

Train a small coding model with [GRPO](https://arxiv.org/pdf/2402.03300) and Hugging Face [TRL](https://huggingface.co/docs/trl/main/en/grpo_trainer) on [Modal](https://modal.com).

Based on the official example: [Train a model to solve coding problems using GRPO and TRL](https://modal.com/docs/examples/grpo_trl).

## Prerequisites

1. A [Modal](https://modal.com) account with GPU access (the example uses **H100**).
2. The [Modal CLI](https://modal.com/docs/guide) installed locally.
3. A [Weights & Biases](https://wandb.ai) account (training logs to W&B).

## Setup

Install [uv](https://docs.astral.sh/uv/), then sync the local environment (Modal CLI only — training deps run in the Modal image):

```bash
uv sync
uv run modal setup
```

Create a Modal secret named `wandb-secret` with your W&B API key:

```bash
uv run modal secret create wandb-secret WANDB_API_KEY=<your-wandb-api-key>
```

## Training

Basic training (1× H100):

```bash
uv run modal run --detach grpo_trl.py::train
```

Faster rollout generation with vLLM — **server mode** (2× H100):

```bash
uv run modal run --detach grpo_trl.py::train_vllm_server_mode
```

vLLM **colocate mode** (1× H100, shares GPU memory with training):

```bash
uv run modal run --detach grpo_trl.py::train_vllm_colocate_mode
```

Checkpoints are written to the Modal volume `example-grpo-trl-checkpoints` at `/models`.

The default run uses 128 dataset rows and `max_steps=5` for a quick smoke test. Increase or remove those limits in `start_grpo_trainer()` for a full run.

## Serving the trained model

After training completes, deploy an OpenAI-compatible vLLM endpoint:

```bash
uv run modal deploy grpo_trl.py
```

Then query the endpoint (replace `<url>` with your deployment URL):

```bash
curl -X POST <url>/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "user", "content": "Write a Python function that returns the sum of a list."}
    ],
    "temperature": 0.7
  }'
```

## What this does

- **Dataset:** [OpenCoder-LLM/opc-sft-stage2](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2) (`educational_instruct`)
- **Model:** [Qwen/Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)
- **Reward:** Generated code is run in a [Modal Sandbox](https://modal.com/docs/guide/sandbox) with dataset assert tests; reward is `1` on success, `0` otherwise.
