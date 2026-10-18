# RTX Spark N1X 128 GB: vLLM owner runs

This folder holds the evidence for the six `rtx-spark-n1x-128gb` recipes. The registry can't rent this card, so the runs were done on the owner's machine:

- **Card:** NVIDIA RTX Spark N1X. nvidia-smi names it `NVIDIA RTX Spark N1X (6144-core Blackwell RTX GPU)` and reports its memory as `[N/A]`: 128 GiB unified with the host, 127.6 GiB online, 119.5 GiB `MemTotal`.
- **Host:** aarch64, 20 cores (Cortex-X925), a laptop running a desktop session. Weights sit on btrfs (`compress=zstd:3`).
- **Software:** Linux 7.2.8, NVIDIA driver 615.78.08, Docker 29.8.2 with CDI devices from nvidia-container-toolkit 1.20.1.

## Runs

Each recipe has one proof, from `lab.py try --on endpoint`, with all six gates passing. Its `log` hash matches its record in `data/lab-runs/rtx-spark-n1x-128gb.*`, all from 2026-10-15, 03:46–05:53 UTC.

**qwen3.8-27b-awq-bf16-int4 is borderline on speed, and both its runs are here.** The same eager launch was gated twice. At 03:25 (`…T032530`) it passed every gate except speed, at 14.9 tok/s. At 05:53 (`…T055305`) it passed everything at 15.7 tok/s; that's the run its proof cites. Over the whole answer it decodes about 32 tok/s. The gate's 30-second window from the first token catches the start of the answer, where DFlash accepts about 2 draft tokens per step (6–14 tok/s); later, acceptance rises to 6–7 (30–40 tok/s). The first run was started by hand without the memory log.

Each recipe is the GB10 recipe's settings with two changes:

- **`mem` 0.7 instead of 0.8.** On unified memory, vLLM compares the request with free host memory. The desktop holds about 22 GiB, so 0.8 (95.6 GiB) refuses to start.
- **`--safetensors-load-strategy eager` added to `extra`.** vLLM's default mmap'd loading copied weights into GPU memory at about 60 MB/s here, even with the files in page cache. A 27 GB checkpoint took 26.5 min, past the plugin's 30-minute start deadline. Eager loads the same checkpoint in seconds.

Each launch ran the way the Local AI plugin runs it:

- `docker run --security-opt no-new-privileges --device nvidia.com/gpu=<uuid> --shm-size <launch shm>`
- the plugin's verified weights mounted read-only at `/models`
- the args `lab.py` renders for the recipe

Before each run, the rendered args were checked to equal, argument for argument, the ones the plugin's export serves for the card, plus the eager flag. The gates ran against the engine's bridge address. The plugin's gateway would have cut the 213K-token context gate off at its 600 s upstream timeout.

## Later runs

**`qwen3.6-35b-a3b.vllm.256k` (2026-10-18)** is NVIDIA's NVFP4 checkpoint with its own MTP heads (no draft to fetch) at `mem` 0.6. The first attempt at 0.7 was stopped by the memory watchdog at 2.8 GiB `MemAvailable`: eager loading holds the checkpoint in host memory while it is copied to the GPU, and that morning's desktop held about 28 GiB, not 22. At 0.6 (71.7 GiB) the machine kept at least 11.5 GiB free throughout.

**`qwen3.8-27b.vllm.256k` (2026-10-18)** is NVIDIA's NVFP4 checkpoint with the same DFlash draft as the AWQ builds, at `mem` 0.6: 36.9 tok/s against the FP8 build's 23.8 at the same AA index. `MemAvailable` dipped to 4.1 GiB for ten seconds while vLLM captured CUDA graphs (GPU allocation 92 GiB, above the 71.7 GiB share, during start only); it held 11+ GiB otherwise. The start took 16 minutes with a 25 GB download running beside it on the same disk.

**`nemotron-3.5-bolt-74b-a9b.llama.cpp.256k` (2026-10-18)** is NVIDIA's 24 GB Bolt GGUF with its image projector and the LiLiCorr DFlash drafter, on llama.cpp's own CUDA 13 image (`server-cuda13-b11718`, arm64), with NVIDIA's RTX Spark flags and none of a host's. It serves 10 s after `docker run` and holds 35 GiB at most, so the machine kept 60 GiB free throughout.

**`nemotron-3.5-super-vl.llama.cpp.128k` (2026-10-18)** is ggml-org's NVFP4 GGUF of NVIDIA's Super VL 120B-A12B, with its Q8_0 image projector and LiLiCorr drafter, at 128K context with a q8_0 cache (eight attention layers, about 1 GiB for the whole window). It passes the speed gate by 0.7 tok/s: the first 30 s of an answer ran at 15.7 tok/s, a 2,112-token answer at 19.8. It holds 83.6 GiB, so it needs the browser and other heavy apps closed; 15.3 GiB stayed free here.

**`qwen3.8-flash-next.llama.cpp.256k` (2026-10-18)** is ISTA-DASLab's GSQ-RCO Q2_0 GGUF of Qwen3.8-Flash-Next (2.40 bpw; the card gives AIME25 96.67 and GPQA-Diamond 89.39 against the BF16 base's 100.00 and 91.92) with its BF16 image projector and unsloth's Q8_0 MTP head as a draft, read whole (`--load-mode none --lazy-mode off`), so its 28.8 GB n-gram table is resident.

The same repository's IQ3_S build passed every gate first (31.0 tok/s, 300 tok/s prefill) with the n-gram table memory-mapped and read on demand (`--load-mode mmap --lazy-mode on`, 55 GB resident). It is not exported: a memory-mapped load reaches the GPU at under 50 MB/s on this machine, as vLLM's mmap loading did above, so it took 19.5 min to start once and over 40 min the next time, against the plugin's 30-minute limit. CPU reads of the same file run at 1.9 GB/s (`pread`) and 1.7 GB/s (touching a mapping), so the slow step is the GPU copy out of mapped pages. Read whole, Q2_0 serves in under a minute.

**Ornith-1.5-35B-A3B and LFM2.5-24B-A2B (2026-10-18)** run whole from their Q8_0 GGUFs on llama.cpp b11718 and leave the desktop 50 GiB or more.

**Not exported, 2026-10-18:** Mellum2.1-12B-A2.5B-Thinking (Q8_0) passed load, chat, reasoning and tools, then on the context gate read the 106K-token prompt in 5 min and thought for 40 min at 3–4 tok/s without answering; the run was stopped. MiniCPM-V 4.7 35B-A3B (Q5_0 with its BF16 DSpark draft at 15 tokens) decoded at 5.0 tok/s and failed the reasoning, context and speed gates.

## Files

- **`memwatch/<model>.csv`:** one per launch, every 5 s from `docker run` until the container was removed.
  - Columns: `t_s` (seconds since `docker run`), `mem_available_mib` (host `MemAvailable`), `gpu_used_mib` (the sum of `nvidia-smi --query-compute-apps=used_memory`, which reports per-process allocation on this iGPU).
  - Host and GPU share the same memory, so the two columns overlap: the engine's allocation is also missing from `MemAvailable`.

## Summary

`MemAvailable` covers the whole machine, including the desktop session.

| recipe | decode tok/s | prefill tok/s | context prompt | weight load | GPU allocation peak (MiB) | host MemAvailable start → min (GiB) |
|---|---|---|---|---|---|---|
| `gemma-4-12b-it-awq-int4.vllm.128k` | 47.7 | 546 | 106258 | 21 s | 81760 | 98.2 → 11.4 |
| `gemma-4-26b-a4b-it-awq-4bit.vllm.256k` | 64.0 | 325 | 213576 | 35 s | 78065 | 97.7 → 11.2 |
| `gemma-4-26b-a4b-it-awq-8bit.vllm.256k` | 52.1 | 327 | 213576 | 41 s | 87989 | 96.9 → 9.6 |
| `qwen3.8-27b-awq-fp8.vllm.256k` | 23.8 | 329 | 213612 | 39 s | 85961 | 97.1 → 11.0 |
| `gemma-4-31b-it-awq-4bit.vllm.256k` | 28.3 | 82 | 213576 | 30 s | 85354 | 97.1 → 8.2 |
| `qwen3.8-27b-awq-bf16-int4.vllm.256k` | 15.7 | 330 | 213612 | 72 s | 87115 | 97.3 → 9.7 |
| `qwen3.6-35b-a3b.vllm.256k` (NVFP4, MTP 2, `mem` 0.6) | 103.0 | 1184 | 213570 | 78 s | 62656 | 90.7 → 11.5 |
| `qwen3.8-27b.vllm.256k` (NVIDIA NVFP4, DFlash, `mem` 0.6) | 36.9 | 506 | 213612 | 39 s | 92425 | 89.6 → 4.1 |
| `nemotron-3.5-bolt-74b-a9b.llama.cpp.256k` (NVIDIA 24 GB GGUF, LiLiCorr draft, llama.cpp b11718) | 33.7 | 590 | 213575 | 10 s to serve | 35100 | 94.4 → 60.0 |
| `nemotron-3.5-super-vl.llama.cpp.128k` (ggml-org NVFP4 GGUF, LiLiCorr draft, q8_0 KV) | 15.7 | 477 | 106257 | 57 s to serve | 83588 | 93.7 → 15.3 |
| `qwen3.8-flash-next.llama.cpp.256k` (ISTA-DASLab GSQ-RCO Q2_0 GGUF, unsloth Q8_0 MTP draft, q8_0 KV) | 37.8 | 366 | 213612 | 59 s to serve | 52898 | 96.9 → 19.2 |
| `ornith-1.5-35b-a3b.llama.cpp.256k` (Q8_0 GGUF, built-in MTP 3) | 46.9 | 640 | 213570 | 32 s to serve | 48806 | 94.9 → 50.6 |
| `lfm2.5-24b-a2b.llama.cpp.128k` (Q8_0 GGUF) | 71.5 | 1149 | 91882 | 21 s to serve | 32378 | 94.7 → 66.7 |

Load is vLLM's "Model loading took". The plugin's whole start, from `docker run` to the API answering, took 182–288 s for each.
