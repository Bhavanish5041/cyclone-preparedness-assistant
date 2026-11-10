# Local Language Model Setup: Qwen2.5-0.5B-Instruct via Ollama

## 1. Overview

CYCLONE INTEL employs **Qwen2.5-0.5B-Instruct** running locally through **Ollama** on Ubuntu Linux. This configuration fulfills all production decision support synthesis without paid APIs, external network dependencies, or dedicated NVIDIA GPU hardware.

### Hardware Footprint
- **Host CPU:** Intel Core i5-13500H (12 cores, 16 threads, 4.7 GHz Max Turbo)
- **Host RAM:** 16 GB DDR5 (system allocated, ~1.5 GB allocated to complete stack)
- **Model Architecture:** Dense decoder-only Transformer (0.49 Billion parameters)
- **Quantization:** Q4_K_M (4-bit quantization)
- **Disk Storage:** 397 MB model weights blob in `/snap/ollama/common/.ollama/models/blobs/`
- **Inference RAM:** ~618 MB RSS resident memory during CPU inference
- **Latency:** ~1.2s to 3.8s on CPU for 100-250 token synthesis

---

## 2. Ollama Daemon Configuration

The Ollama daemon is managed via Ubuntu Snap:

```bash
# Verify daemon service status
systemctl status snap.ollama.ollama.service

# Check local listening socket (strictly loopback)
ss -tulpn | grep 11434
# Output: tcp LISTEN 127.0.0.1:11434
```

### Model Inspection
```bash
ollama list
# Output:
# NAME                     ID              SIZE      MODIFIED
# qwen2.5:0.5b-instruct    c5396e06af29    397 MB    Active
```

---

## 3. Concurrency and CPU Bounding

Because inference runs on a laptop CPU without dedicated GPU tensor cores, concurrent LLM requests could saturate CPU execution queues and degrade response latency.

To prevent this, `src/assistant/llm/client.py` implements an explicit asynchronous semaphore:

```python
self._semaphore = asyncio.Semaphore(1)

async def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.0) -> str:
    async with self._semaphore:
        # Executes exactly 1 model generation pass at a time
        # Queues subsequent turns safely
        ...
```

### Timeout Management
- Default timeout configured in `.env`: `OLLAMA_TIMEOUT_SECONDS=60.0`
- If Ollama fails or times out, the assistant automatically triggers **deterministic fallback formatting**, returning the authoritative calculations immediately with `is_fallback=True` so that users never experience broken requests.

---

## 4. Verification and Testing

You can verify the local model pipeline directly via CLI:

```bash
# 1. Test raw Ollama generate endpoint
curl -s http://127.0.0.1:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5:0.5b-instruct", "prompt": "State the 3 coastal districts of Karnataka.", "stream": false}'

# 2. Test via Python AsyncClient
.venv/bin/python -c "
import asyncio
from src.assistant.llm.client import get_default_llm_client

async def test():
    client = get_default_llm_client()
    print('Provider:', client.provider_name)
    print('Model:', client.model)
    res = await client.generate('Explain coastal vulnerability in 1 sentence.')
    print('Output:', res)

asyncio.run(test())
"
```
