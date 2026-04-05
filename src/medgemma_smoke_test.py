"""
Smoke test: load Med-Gemma 4B and run one text prompt.
Verifies: CUDA access, model loading, tokenization, generation.
No images needed — text-only test.
"""
import torch
import time

print("=" * 60)
print("Med-Gemma 4B Smoke Test")
print("=" * 60)

# 1. Check CUDA
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version:    {torch.version.cuda}")
    print(f"GPU device:      {torch.cuda.get_device_name(0)}")
    print(f"GPU memory:      {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"bfloat16 support: {torch.cuda.is_bf16_supported()}")
else:
    print("ERROR: No GPU detected. Exiting.")
    exit(1)

# 2. Determine dtype
# RTX 3090/4090 support bfloat16. Older GPUs (V100, 1080, 2080) do not.
if torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
    print(f"\nUsing bfloat16 (native support)")
else:
    dtype = torch.float16
    print(f"\nUsing float16 (bfloat16 not supported on this GPU)")

# 3. Load model
print("\nLoading model (this takes 1-3 minutes)...")
t0 = time.time()

from transformers import AutoProcessor, AutoModelForImageTextToText

model_id = "google/medgemma-4b-it"

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

load_time = time.time() - t0
print(f"Model loaded in {load_time:.1f}s")
print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# 4. Run a simple text-only prompt
print("\nRunning inference...")
t0 = time.time()

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful medical assistant."}]
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": (
            "What CBC findings are commonly associated with "
            "rheumatoid arthritis? List the top 3 most relevant."
        )}]
    }
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt"
).to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=300, do_sample=False)

response = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
gen_time = time.time() - t0

print(f"\nResponse ({gen_time:.1f}s):")
print("-" * 40)
print(response)
print("-" * 40)

print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print("\nSMOKE TEST PASSED")