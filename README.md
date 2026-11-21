# Aegis Router v1

**Sub-6ms, zero-egress LLM routing, prompt-injection defense, and tool dispatch in a single forward pass.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Weights-yellow.svg)](https://huggingface.co/AkshaXA/aegis-router-v1)
[![Live Demo](https://img.shields.io/badge/Demo-WebGPU_Zero_Egress-green.svg)]([https://devxa2602.github.io/AegisRouter/aegisRouter.html])

![Aegis Router T4 Benchmark GIF](assets/t4_demo.gif)
*NVIDIA T4 GPU (ONNX Runtime FP16, BS=1): 5.91 ms p50 | 157.3 req/sec*

---

## Try It Locally (Zero Data Egress)

Run the full ONNX FP16 model **100% inside your browser** via WebGPU/WASM—no signup, no backend, zero API calls:

👉 **[Launch Zero-Egress Browser Demo](https://devxa2602.github.io/AegisRouter/aegisRouter.html)**

*Open Chrome DevTools → Network tab after loading the ~300 MB weights to verify zero network requests leave your machine. Browser WebGPU runs at ~100 ms; T4 server latency is **5.91 ms p50**.*

---

## Why Aegis Router?

Production LLM gateways usually force a bad trade-off:
* **Rule-based proxies (LiteLLM/Kong):** Fast, but lack semantic understanding for model tiering or adversarial prompt detection.
* **Hosted routers:** Add 100–500 ms of network latency and send raw user prompts (and PII) to a third-party API.
* **LLM-as-a-Router (Haiku/GPT-4o-mini):** Burns tokens and adds 400–800 ms before your actual LLM call even starts.

**Aegis Router** runs inside your own VPC using a single 149M-parameter `ModernBERT-base` encoder pass (zero tokens generated) with three multi-task heads:
1. **Calibrated Model Tiering (`SLM` / `Mid` / `Frontier`):** Confidence threshold set at `0.85`—anything below automatically escalates to Frontier so routing errors fail safe.
2. **Prompt-Injection Guardrail:** Hard-blocks adversarial inputs before downstream LLM spend.
3. **5-Class Tool Dispatch:** Routes directly to `DIRECT_CHAT`, `CODE_OPS`, `RAG_RETRIEVAL`, `STRUCTURED_SQL`, or `EXTERNAL_API`.

---

## Benchmarks

Evaluated on an **NVIDIA T4 GPU** (ONNX Runtime FP16, `batch_size=1`):

| Benchmark | Aegis Router v1 |
| :--- | :--- |
| **p50 Latency** | **5.91 ms** |
| **p95 / p99 Latency** | 8.44 ms / 9.08 ms |
| **Throughput (BS=1)** | **157.3 req/sec** |
| **Prompt-Injection Guardrail** | **99.07% Acc** (0.9841 F1) |
| **5-Class Tool Dispatch** | **98.75% Acc** (0.9878 Macro-F1) |
| **Expected Calibration Error (ECE)** | **0.0572** |

![Aegis Router T4 Throughput Benchmark](assets/t4_benchmark.gif)
*Reproducible Kaggle T4 run (`N=500`, `BS=1`) — see [`notebooks/aegis_t4_benchmark.ipynb`](notebooks/aegis_t4_benchmark.ipynb)*

---

## Quickstart

### 1. Download Weights
Weights are hosted on Hugging Face: [AkshaXA/aegis-router-v1](https://huggingface.co/AkshaXA/aegis-router-v1)

```bash
pip install onnxruntime-gpu transformers huggingface_hub
2. Run InferencePythonfrom huggingface_hub import hf_hub_download
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

model_path = hf_hub_download("AkshaXA/aegis-router-v1", "aegis_router_v1_fp16.onnx")
tokenizer = AutoTokenizer.from_pretrained("AkshaXA/aegis-router-v1")
session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

inputs = tokenizer("Write a SQL query to find monthly active users.", return_tensors="np", truncation=True)
outputs = session.run(None, {
    "input_ids": inputs["input_ids"].astype(np.int64),
    "attention_mask": inputs["attention_mask"].astype(np.int64),
})
```
### Roadmap & EnterpriseOpen-Source (Apache 2.0): 
* **Aegis Router v1 weights (149M), ONNX FP16 graph, basic FastAPI proxy, and WebGPU demo**.
* **Coming in v2 (Aegis 300M)**: Custom 3:1 Mamba + FlashAttention-2 hybrid backbone for linear $O(N)$ routing and indirect prompt-injection defense across 64K–128K multi-turn agent trajectories.
* **Enterprise Control Plane**: Custom LoRA heads for proprietary tool schemas, RBAC, and SOC2 audit logging. Interested in a free in-VPC design partnership? Reach out at akshayrawate@gmail.com. LicenseApache 2.0 — see LICENSE for details.
