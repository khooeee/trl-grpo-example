# Train a model to solve coding problems using GRPO and TRL on Modal.
# Docs: https://modal.com/docs/examples/grpo_trl

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

import modal

app: modal.App = modal.App("example-grpo-trl")

image: modal.Image = modal.Image.debian_slim().uv_pip_install(
    "trl[vllm]==0.28.0",
    "vllm==0.12.0",
    "transformers==4.57",
    "datasets==3.5.1",
    "wandb==0.17.6",
)

with image.imports():
    from datasets import Dataset, load_dataset
    from trl import GRPOConfig, GRPOTrainer

MODELS_DIR = Path("/models")
checkpoints_volume: modal.Volume = modal.Volume.from_name(
    "example-grpo-trl-checkpoints", create_if_missing=True
)


@app.function()
def compute_reward(completion: str, testcase: Sequence[str]) -> int:
    sb, score = None, 0
    sb: modal.Sandbox = modal.Sandbox.create(app=app)
    code_to_execute: str = get_generated_code_and_test_cases(completion, testcase)

    try:
        p = sb.exec("python", "-c", code_to_execute, timeout=30)
        p.wait()
        return_code = p.returncode
        if return_code == 0:
            score = 1
    except Exception as e:
        print(e)
    finally:
        sb.terminate()
        return score


def get_generated_code_and_test_cases(completion: str, testcase: Sequence[str]) -> str:
    if "```python" in completion:
        start_idx: int = completion.find("```python") + len("```python")
        end_idx: int = completion.find("```", start_idx)
        if end_idx != -1:
            code: str = completion[start_idx:end_idx].strip()
        else:
            code: str = completion[start_idx:].strip()
    else:
        code: str = completion.strip()

    test_cases: str = "\n".join(testcase)
    full_code: str = f"{code}\n\n{test_cases}"
    return full_code


def reward_helper_function(
    completions: Sequence[str], testcases: Sequence[Sequence[str]], **kwargs: object
) -> Iterable[int]:
    return compute_reward.starmap(zip(completions, testcases))


def start_grpo_trainer(use_vllm: bool = False, vllm_mode: str | None = None) -> None:
    dataset: Dataset = load_dataset(
        "OpenCoder-LLM/opc-sft-stage2", "educational_instruct", split="train"
    )
    dataset = dataset.rename_column("instruction", "prompt")
    dataset = dataset.rename_column("testcase", "testcases")
    dataset = dataset.select(range(128))  # To simplify testing.
    training_args: GRPOConfig = GRPOConfig(
        output_dir=str(MODELS_DIR),
        report_to="wandb",
        use_vllm=use_vllm,
        vllm_mode=vllm_mode,
        save_steps=1,
        max_steps=5,  # To simplify testing. Remove for production use cases.
    )
    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_helper_function,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 24,
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={"/models": checkpoints_volume},
)
def train() -> None:
    start_grpo_trainer()


@app.function(
    image=image,
    gpu="H100:2",
    timeout=60 * 60 * 24,
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={str(MODELS_DIR): checkpoints_volume},
)
def train_vllm_server_mode() -> None:
    env_copy = os.environ.copy()
    env_copy["CUDA_VISIBLE_DEVICES"] = "0"

    subprocess.Popen(
        ["trl", "vllm-serve", "--model", "Qwen/Qwen2-0.5B-Instruct"],
        env=env_copy,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    start_grpo_trainer(use_vllm=True, vllm_mode="server")


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 24,
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={"/models": checkpoints_volume},
)
def train_vllm_colocate_mode() -> None:
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    start_grpo_trainer(use_vllm=True, vllm_mode="colocate")


VLLM_PORT: int = 8000


def get_latest_checkpoint_file_path() -> str:
    checkpoint_dirs = [
        d.name
        for d in MODELS_DIR.iterdir()
        if d.is_dir() and re.match(r"^checkpoint-(\d+)$", d.name)
    ]
    if not checkpoint_dirs:
        raise FileNotFoundError("No checkpoint directories found in models dir")
    latest_checkpoint_index = max(
        int(re.match(r"^checkpoint-(\d+)$", d).group(1)) for d in checkpoint_dirs
    )
    return str(MODELS_DIR / f"checkpoint-{latest_checkpoint_index}")


vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "vllm==0.12.0",
        "flashinfer-python==0.5.3",
        extra_index_url="https://download.pytorch.org/whl/cu128",
        extra_options="--index-strategy unsafe-best-match",
    )
    .env({"VLLM_USE_V1": "1"})
)

vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


@app.function(
    image=vllm_image,
    gpu="H100",
    scaledown_window=15 * 60,
    timeout=10 * 60,
    volumes={"/root/.cache/vllm": vllm_cache_vol, MODELS_DIR: checkpoints_volume},
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * 60)
def serve() -> None:
    latest_checkpoint_file_path = get_latest_checkpoint_file_path()

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        latest_checkpoint_file_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
    ]
    subprocess.Popen(" ".join(cmd), shell=True)
