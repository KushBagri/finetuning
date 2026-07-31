# Fine-Tuning Qwen2.5-7B-Instruct into a Sarcastic Chatbot with QLoRA

This README walks through **every single piece** of the `finetune.ipynb` notebook — not just what each line does, but *why* it exists, what would break without it, and the underlying concept it represents. Read top to bottom; each section builds on the last.

---

## 1. The Big Picture — What Is Fine-Tuning, and Why This Approach?

Qwen2.5-7B-Instruct is a **7 billion parameter** language model. Think of "parameters" as knobs — 7 billion tiny dials that together encode everything the model knows about language, facts, and reasoning. It's already been trained to be a helpful, neutral assistant.

Your goal: turn that neutral assistant into a **sarcastic, satirical** one, *without* retraining all 7 billion knobs from scratch (that would take a data-center, not a laptop).

The solution here uses two ideas stacked together:

- **LoRA** (Low-Rank Adaptation) — instead of adjusting all 7 billion knobs, you freeze the original model and attach small, trainable "patch" layers next to it. You only train the patches.
- **QLoRA** (Quantized LoRA) — you additionally shrink the frozen base model down to 4-bit precision so it fits in a tiny amount of GPU memory, while the small LoRA patches are trained in full precision on top.

**Analogy:** Imagine a giant, finely tuned orchestra (the base model) that already knows how to play beautifully. You don't want to retrain every musician. Instead, you hire a small group of "mood filters" — a few extra musicians who sit on top and reinterpret the existing music with a sarcastic twist. The orchestra stays frozen (nobody re-learns their instrument); only the filter musicians (LoRA layers) rehearse and change. And to save money on the venue, you photocopy the orchestra's sheet music in a compressed, low-resolution format (4-bit quantization) — good enough to play from, much cheaper to store.

This is why the notebook comments say **~3.5–4 GB VRAM** and **~3–4 GB RAM** — this whole trick is specifically engineered to let a 7B model, which would normally need ~28 GB+ in full precision, run on something like a single consumer GPU (e.g., a laptop RTX card or a free Colab GPU).

---

## 2. Installation Cell

```python
# !pip install trl bitsandbytes
```

This is commented out (a `#` in front) — meaning it's there as a *reminder*, not something that runs automatically. The notebook's second cell also lists a fuller install command:

```
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft trl bitsandbytes datasets accelerate
```

Here's what each package is actually responsible for:

| Package | Role |
|---|---|
| `torch` | The core deep learning engine — actually does the math (matrix multiplications, gradients, GPU operations). Everything else sits on top of it. |
| `transformers` | Hugging Face's library for loading pretrained models (like Qwen2.5) and tokenizers, and running them. |
| `peft` | "Parameter-Efficient Fine-Tuning" — this is the library that implements **LoRA** itself. |
| `trl` | "Transformer Reinforcement Learning" — despite the name, it's used here for its `SFTTrainer`, a convenience wrapper for **Supervised Fine-Tuning** (training on labeled input→output examples). |
| `bitsandbytes` | Implements the **4-bit quantization** math (and 8-bit optimizers) that make QLoRA possible on small GPUs. |
| `datasets` | Hugging Face's library for loading and processing training data efficiently (handles files too big for memory, batching, splitting, etc.). |
| `accelerate` | Handles device placement — deciding what goes on the GPU vs CPU, and coordinating multi-GPU/distributed setups if you had them. |

The `cu121` in the torch install URL means "CUDA 12.1" — this must match the CUDA version your GPU drivers support, or torch won't be able to talk to the GPU at all.

---

## 3. Imports Cell

```python
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import warnings
warnings.filterwarnings('ignore')
```

Breaking down the less-obvious imports:

- **`AutoModelForCausalLM`** — "Auto" means it inspects the model name/config and automatically picks the right architecture class (Qwen2.5 uses a Llama-like transformer decoder under the hood; you don't have to know that detail — `Auto` figures it out). "CausalLM" = **Causal Language Model**, meaning the model predicts the *next* token given only the tokens *before* it (it can't peek ahead) — this is the standard architecture for chat models.
- **`AutoTokenizer`** — converts text ↔ numbers. Models don't read words; they read integer IDs from a fixed vocabulary. The tokenizer is the translator in both directions.
- **`BitsAndBytesConfig`** — the settings object that tells `transformers` *how* to quantize the model into 4-bit when loading it.
- **`LoraConfig`** — settings object describing the shape/size of the LoRA "patch" layers.
- **`prepare_model_for_kbit_training`** — a helper function that does necessary housekeeping *specifically* required when you're about to train on top of a k-bit (here, 4-bit) quantized model (more on this below).
- **`SFTTrainer` / `SFTConfig`** — the training loop and its configuration, from `trl`. This is a wrapper around Hugging Face's generic `Trainer` that's specialized for instruction/chat-style supervised fine-tuning.
- **`warnings.filterwarnings('ignore')`** — purely cosmetic; suppresses Python warning spam (deprecated function notices, etc.) so the notebook output stays readable.

---

## 4. Paths Cell

```python
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATA_PATH = "./drive/MyDrive/SaasLM/sarcasm_dataset.jsonl"
OUTPUT_DIR = "./drive/MyDrive/SaasLM/sarcastic-qwen-lora v1.0"
```

- `MODEL_ID` is a Hugging Face Hub identifier. When you pass this to `AutoModelForCausalLM.from_pretrained(...)`, it downloads the model weights and config directly from Hugging Face's servers (and caches them locally after the first run).
- `DATA_PATH` and `OUTPUT_DIR` point into a `./drive/MyDrive/...` path — this is the tell-tale sign this notebook is meant to run in **Google Colab** with Google Drive mounted, so your dataset and trained adapter persist across sessions (Colab wipes local disk when the runtime disconnects).
- Note: `OUTPUT_DIR` has a **space** in it (`lora v1.0`). This works on Linux/Colab filesystems but is worth flagging — spaces in paths can cause subtle bugs in some shell-based tooling downstream (e.g., if you ever `zip`/`scp` this folder from a raw shell command without quoting the path).

---

## 5. Dataset Loading

```python
dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.train_test_split(test_size=0.05, seed=42)
```

### The expected data format

Each line of `sarcasm_dataset.jsonl` is one JSON object:

```json
{"messages": [
   {"role": "user", "content": "How do I boil an egg?"},
   {"role": "assistant", "content": "Ah yes, humanity's greatest unsolved mystery..."}
]}
```

This is called **JSONL** (JSON Lines) — one valid JSON object per line, rather than one giant array. It's the standard format for chat fine-tuning datasets because it can be streamed line-by-line without loading a huge file into memory all at once (this matters even more at larger scale than the 500–2000 examples mentioned here).

The `"messages"` structure mirrors exactly how a chat model is prompted at inference time: a list of turns tagged by `role` (`user`, `assistant`, sometimes `system`). This is deliberate — SFT works by showing the model realistic *conversations* so it learns the target behavior (sarcasm) in the same shape it will be used later.

**Why 500–2000 examples is "enough":** you are not teaching the model new facts or new reasoning ability — the base model already knows how to boil an egg, write code, etc. You're teaching it a **style** (sarcasm). Style transfer needs far fewer examples than teaching new knowledge, but each example needs to be *consistent* in voice — a wobbly, inconsistent dataset teaches the model an inconsistent, "confused" personality.

### The train/test split

```python
dataset.train_test_split(test_size=0.05, seed=42)
```

This carves off 5% of your examples as a **holdout/validation set** that the model never trains on. Its purpose: after each round of training, you can check the model's loss on data it *hasn't* memorized, to see if it's actually generalizing the "sarcastic style" or just memorizing your training examples verbatim (overfitting).

**Analogy:** think of it like studying for an exam using practice questions (`train`), but keeping a few practice questions hidden (`test`) that you only look at right before the real exam — if you do well on those too, you've actually learned the material, not just memorized the specific practice questions.

`seed=42` makes the split **reproducible** — every time you rerun this cell, you get the *exact same* random split, rather than a different random 5% each time. This matters for fair comparisons if you tweak hyperparameters and rerun.

---

## 6. Loading the Base Model in 4-bit (This Is the Heart of QLoRA)

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

Let's unpack **quantization** properly, because this is the single biggest reason this fine-tune fits on a small GPU at all.

### What is quantization?

Every number (weight) in the model is normally stored as a **32-bit** or **16-bit floating point number** — think of this as a very precise decimal, like `0.7391827451...`. Quantization rounds these numbers down to a much coarser representation — here, **4 bits** per weight, meaning each weight can only take one of 16 possible values.

**Analogy:** Imagine describing colors. Full precision (16/32-bit) is like having access to millions of shades of paint. 4-bit quantization is like being handed a box of 16 crayons and having to pick the *closest* crayon color to each shade in the original painting. You lose some nuance, but the picture is still clearly recognizable — and the box of 16 crayons is far cheaper and smaller to carry around than millions of paint cans.

Since a weight now takes 4 bits instead of 16 or 32, the model's memory footprint shrinks roughly **4–8x**. That's what turns a model needing ~28 GB into one needing ~4–5 GB.

### Breaking down each setting:

- **`load_in_4bit=True`** — the master switch: load the model's weights compressed into 4-bit representation instead of the usual 16/32-bit.
- **`bnb_4bit_quant_type="nf4"`** — "NF4" stands for **4-bit NormalFloat**. This is a specific *smart* way of choosing which 16 values those 4 bits represent. Instead of spacing the 16 buckets evenly (like a plain ruler), NF4 spaces them to match the *actual statistical distribution* of neural network weights, which cluster tightly around zero in a bell-curve (normal distribution) shape. More "crayons" (finer resolution) are allocated where most of the weights actually live, fewer where they're rare. This meaningfully reduces the "rounding error" versus naive 4-bit quantization.
- **`bnb_4bit_compute_dtype=torch.bfloat16`** — this is subtle and important: the weights are *stored* in 4-bit, but when the GPU actually needs to do math (multiply weights by activations), bitsandbytes **decompresses them on-the-fly** into `bfloat16` (a 16-bit floating format) to actually run the computation, then discards the decompressed copy. `bfloat16` ("Brain Float 16") has the same exponent range as 32-bit float (so it doesn't overflow/underflow easily) but a shorter mantissa (less precision) — it's the sweet spot most modern GPUs (Ampere/Hopper NVIDIA architectures and newer) are hardware-accelerated for. The comment `# use torch.float16 if your GPU lacks bf16` exists because older GPUs (pre-Ampere, e.g., many older consumer cards) don't have bf16 hardware support and would need the older fp16 format instead.
- **`bnb_4bit_use_double_quant=True`** — a clever extra trick: quantization itself needs a small amount of extra metadata per block of weights (called "quantization constants"). Double quantization *also* quantizes those metadata constants, squeezing out a bit more memory savings (typically ~0.4 GB less on a 7B model) essentially for free.

### Loading the tokenizer and model

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="sdpa",
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
```

- **`device_map="auto"`** — lets the `accelerate` library decide automatically which parts of the model go on GPU vs CPU (and split across multiple GPUs if present). For a single-GPU QLoRA setup, this typically just means "put it all on the GPU."
- **`attn_implementation="sdpa"`** — chooses **Scaled Dot-Product Attention**, PyTorch's built-in, memory-efficient, fused implementation of the attention mechanism (the core operation transformers use to let tokens "look at" each other). It's faster and uses less memory than the naive/manual implementation, without needing extra third-party libraries like FlashAttention.
- **`model.config.use_cache = False`** — this disables the **KV cache** (key/value cache), a feature normally used during *generation* to avoid recomputing attention for tokens you've already processed. It must be turned off during *training* because it conflicts with **gradient checkpointing** (explained below) — you can't cache old activations and also intentionally throw them away to save memory in the same run.
- **`prepare_model_for_kbit_training(model)`** — a `peft` helper that does several pieces of essential housekeeping for training on top of a quantized model:
  1. Casts certain small layers (like LayerNorm) back to full precision for numerical stability, since some operations misbehave in low precision.
  2. Enables gradient computation on the *input embeddings* (normally frozen/no-grad in inference), which is required because with the base model fully frozen, gradients need somewhere to "flow into" before reaching the trainable LoRA layers.
  3. Enables gradient checkpointing compatibility.

Without this preparation step, attempting to backpropagate through a 4-bit quantized, frozen model would either error out or silently fail to train properly.

---

## 7. LoRA Configuration — The Trainable "Patch" Layers

```python
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
```

### How LoRA actually works, mechanically

Normally, fine-tuning a layer means updating its entire weight matrix `W` (which for a 7B model layer might be thousands-by-thousands of numbers). LoRA's insight: instead of updating `W` directly, freeze it, and add a *small* correction:

```
W_new = W_frozen + (A × B)
```

Where `A` and `B` are two much *smaller* matrices whose product approximates the "delta" the model needs to shift its behavior. Only `A` and `B` are trained; `W_frozen` never changes.

**Analogy:** Instead of repainting an entire mural (the frozen weights), you place a semi-transparent colored overlay (the small `A × B` matrices) on top that shifts the mood of the whole piece. The original mural underneath is untouched; you can even peel the overlay off later and get the original mural back exactly.

### Parameter breakdown:

- **`r=16`** — the "rank" of the LoRA update, i.e., how "wide" that overlay is (literally the inner dimension of matrices `A` and `B`). Higher `r` = more expressive capacity to change behavior, but more trainable parameters and more VRAM/compute. `r=16` is a small-to-moderate rank — appropriate for a style-transfer task like sarcasm, where you're nudging *tone*, not injecting large amounts of new knowledge.
- **`lora_alpha=32`** — a scaling factor applied to the LoRA update before adding it to the frozen weights (roughly, the update is scaled by `alpha / r`). It controls *how strongly* the LoRA patch influences the final output relative to the frozen base weights. The common convention (used here) is to set `alpha` to about **2× the rank**, which tends to produce well-balanced training dynamics without needing to retune the learning rate specially.
- **`lora_dropout=0.05`** — during training, 5% of the LoRA pathway's connections are randomly "zeroed out" on each forward pass. This is a regularization technique to prevent the small number of trainable parameters from overfitting to quirks of your (relatively small) dataset.
- **`bias="none"`** — LoRA can optionally also make the bias terms (a simpler additive parameter per neuron) trainable. `"none"` means don't touch them — keep the parameter count and complexity minimal, since biases rarely matter much for this kind of style-transfer task.
- **`task_type="CAUSAL_LM"`** — tells `peft` which architecture pattern to expect internally (as opposed to e.g. sequence classification or seq2seq), so it wires up the LoRA layers correctly for a decoder-only chat model.
- **`target_modules=[...]`** — this is the list of *which specific weight matrices inside each transformer layer* get a LoRA patch attached. In a standard transformer block these are:
  - `q_proj`, `k_proj`, `v_proj`, `o_proj` — the four projection matrices inside the **self-attention** mechanism (Query, Key, Value, and Output projections) — these control how tokens attend to and blend information from each other.
  - `gate_proj`, `up_proj`, `down_proj` — the three matrices inside the **feed-forward (MLP) block** of each layer — these do the "thinking" after attention has gathered context.
  
  Targeting *all seven* of these (rather than just attention, which is an older, more conservative LoRA convention) gives the adapter influence over both the "attention" and "reasoning" parts of every layer — generally producing a stronger, more expressive stylistic shift, at a modest additional parameter cost.

---

## 8. Training Configuration

```python
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    max_length=1024,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    save_steps=20,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    bf16=True,
    optim="paged_adamw_8bit",
    report_to="none",
)
```

Let's go through these in logical groups.

### a) How much data the model sees, and how

- **`num_train_epochs=3`** — the *entire* training dataset is passed through the model **3 times**. For small style-transfer datasets, a handful of epochs is typical — too many epochs on a small dataset risks the model memorizing your exact examples word-for-word instead of generalizing the sarcastic *style*.
- **`per_device_train_batch_size=2`** — how many examples are processed **simultaneously** in one forward/backward pass, per GPU. Note the code comment says *"keep at 1 for the VRAM budget"* — meaning the value `2` here is actually slightly above what the stated 3.5–4 GB budget assumes; if you hit an out-of-memory error, dropping this to `1` is the first fix to try.
- **`gradient_accumulation_steps=4`** — this is the trick that lets you *simulate* a bigger batch size without the memory cost of actually processing a bigger batch at once. The model processes small batches one at a time, computes gradients each time, and **adds them up** across 4 steps before actually updating the weights once. 

  **Analogy:** Imagine you want to carry 8 buckets of water across a yard, but you can only carry 2 at a time. Gradient accumulation is: make 4 trips of 2 buckets each, but only "pour into the tank" (update the weights) after all 4 trips — the *effect* on the tank is the same as if you'd carried all 8 at once, you just never needed the strength (VRAM) to hold 8 simultaneously.
  
  Effective batch size = `per_device_train_batch_size × gradient_accumulation_steps` = `2 × 4 = 8` (as the notebook comment confirms).

- **`max_length=1024`** — the maximum number of tokens (word-pieces) any single training example is allowed to be. Longer conversations get truncated. This caps memory use per example — attention's memory cost grows roughly with the *square* of sequence length, so this is a meaningful lever. 1024 tokens is generous for short Q&A-style sarcastic exchanges but would truncate longer conversations.

### b) Memory-saving tricks

- **`gradient_checkpointing=True`** — normally, during the forward pass, the model stores every intermediate calculation ("activations") in memory so it can use them later during the backward pass (to compute gradients). Gradient checkpointing instead **discards** most of these intermediate activations immediately, and **recomputes** them on-demand during the backward pass when they're actually needed.
  
  **Analogy:** it's the difference between keeping every rough draft of an essay you ever wrote (fast to look back at, but takes up a whole filing cabinet) versus shredding each draft right after using it, and rewriting it again from notes if you ever need to see it later (saves cabinet space, costs you a bit of extra writing time). This trades **compute time** (slower training, since some math is redone) for **memory** (dramatically less VRAM), which is exactly the trade-off needed to fit a 7B model's training in ~4GB.

- **`optim="paged_adamw_8bit"`** — this addresses the **optimizer's** memory cost, which is separate from the model weights themselves. AdamW (the optimization algorithm used to update weights during training) needs to keep two extra numbers ("momentum" and "variance" estimates) *for every single trainable parameter*, normally in 32-bit precision — this can easily double or triple the memory needed just for training bookkeeping. `paged_adamw_8bit` does two things:
  1. Stores those optimizer states in **8-bit** instead of 32-bit (another quantization trick, this time applied to training statistics rather than the model weights).
  2. Uses **paging** — borrowed from how operating systems manage RAM — to automatically shuttle optimizer state between GPU memory and CPU RAM when the GPU starts running low, rather than crashing with an out-of-memory error.

### c) Learning rate and schedule

- **`learning_rate=2e-4`** (i.e., 0.0002) — how big a step the optimizer takes when updating the LoRA weights based on each gradient. This is notably *higher* than typical full-model fine-tuning rates (often ~2e-5), which makes sense because LoRA is training far fewer parameters from scratch (small, randomly-initialized matrices) — they need bigger, faster steps to move into a useful configuration, whereas the huge frozen base model would need much gentler nudges if it were being tuned directly.
- **`lr_scheduler_type="cosine"`** — rather than using one fixed learning rate the whole time, it follows a smooth curve shaped like the top half of a cosine wave: starting high and gradually curving down to near-zero by the end of training. This tends to produce better final results than a constant rate — large steps early on for fast progress, tiny careful steps at the end for fine-tuning precision.
- **`warmup_ratio=0.05`** — for the first 5% of training steps, the learning rate ramps up gradually from 0 to the full `2e-4`, instead of starting at full strength immediately. This avoids destabilizing the model with a large update before it's had a chance to adjust to the new data/gradients (a common cause of training instability/divergence if skipped).

### d) Monitoring, evaluation, and checkpointing

- **`logging_steps=10`** — print training metrics (like current loss) every 10 steps, so you can watch progress live.
- **`eval_strategy="steps"` / `eval_steps=20`** — every 20 training steps, pause and run the model against the held-out **test** split (the 5% carved off earlier) to compute a validation loss — this tells you if the model is generalizing or overfitting.
- **`save_strategy="steps"` / `save_steps=20`** — every 20 steps, save a checkpoint (a snapshot of the current LoRA weights) to disk.
- **`save_total_limit=2`** — only keep the **2 most recent** checkpoints on disk at any time, automatically deleting older ones — this prevents your Google Drive from filling up with dozens of large checkpoint folders over a long training run.
- **`load_best_model_at_end=True`** + **`metric_for_best_model="eval_loss"`** — at the very end of training, instead of keeping whatever the *last* checkpoint happened to be, the trainer looks back across all evaluated checkpoints and reloads whichever one had the **lowest validation loss**. This guards against the last checkpoint actually being slightly *worse* (e.g., if the model started mildly overfitting in the final steps).
- **`bf16=True`** — run the *trainable* computations (the LoRA layers, gradients, etc.) in bfloat16 precision rather than full 32-bit float, for speed and memory savings, consistent with the `bnb_4bit_compute_dtype` chosen above. (The comment notes: switch to `fp16=True` on GPUs that don't support bf16.)
- **`report_to="none"`** — disables sending training logs to any external experiment-tracking service (like Weights & Biases or TensorBoard) — logs just print to the notebook output instead.

### Creating the trainer

```python
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    peft_config=peft_config,
)
```

This assembles everything defined so far — the quantized base model, the training hyperparameters, the two dataset splits, the tokenizer, and the LoRA config — into a single object that knows how to run the full training loop. Internally, `SFTTrainer` will:
1. Automatically apply the tokenizer's **chat template** to each `messages` example (turning the structured JSON into the exact text format — with special tokens marking turn boundaries — that Qwen2.5 expects to see).
2. Tokenize the resulting text into IDs.
3. Wrap the base model with the LoRA adapter layers defined by `peft_config` (this actually happens automatically the moment `peft_config` is passed in — you'll notice the raw `model` object passed in gets transformed into a PEFT-wrapped model under the hood).
4. Run the standard training loop: forward pass → compute loss → backward pass → (accumulate gradients) → optimizer step → repeat.

---

## 9. Training and Saving

```python
trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"LoRA adapter saved to {OUTPUT_DIR}")
```

- **`trainer.train()`** kicks off the entire loop described above, for the full 3 epochs, printing logging/eval output along the way per the config.
- **`trainer.save_model(OUTPUT_DIR)`** — crucially, because you used LoRA, this does **not** save a new 7B-parameter model. It saves only the small `A`/`B` adapter matrices (typically a few tens to a couple hundred MB, versus ~15 GB+ for the full base model in 4-bit, or 28GB+ full precision). This is one of LoRA's biggest practical wins: your fine-tuned "personality" is a small, portable file that can be shared, versioned, and swapped independently of the giant base model.

---

## 10. Reloading the Fine-Tuned Model for Inference

```python
base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto"
)
model = PeftModel.from_pretrained(base, OUTPUT_DIR)
model.eval()
```

To actually *use* the fine-tuned model (potentially in a fresh session, e.g., after restarting the notebook), you need two ingredients again: the original frozen base model (re-downloaded/re-quantized exactly as before), and the small saved LoRA adapter layered on top via `PeftModel.from_pretrained`. This mirrors how the model was constructed during training — LoRA doesn't exist as a standalone model; it's always "base + patch."

`model.eval()` switches the model into **evaluation mode**, which disables training-only behaviors like dropout (recall `lora_dropout=0.05` from earlier — you don't want random connections dropped during actual use, only during training) and batch-normalization-style statistics updates (not directly relevant here, but part of the same switch).

Note: this cell doesn't re-set `model.config.use_cache = False` — and in fact for inference you generally *want* `use_cache=True` (the default), since the KV cache is what makes text generation fast by avoiding recomputation of attention for already-generated tokens. It was only disabled during training due to the gradient checkpointing conflict mentioned earlier.

---

## 11. The Inference/Generation Function

```python
def ask_question(question, temperature = 0.2, max_new_tokens = 500):
  messages = [
      {"role": "user", "content": f"{question}"},
  ]
  inputs = tokenizer.apply_chat_template(
      messages, add_generation_prompt=True, return_tensors="pt"
  ).to(model.device)

  with torch.no_grad():
      out = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=True)
  print(tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
  return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
```

Walking through line by line:

- **`messages = [{"role": "user", "content": question}]`** — wraps the raw question string into the same `messages` structure used in training, so the model receives input in the shape it was fine-tuned to expect.
- **`tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")`** — this converts the structured message list into the exact raw text string Qwen2.5 was trained on (including special marker tokens like `<|im_start|>user` etc., which the tokenizer knows about internally), then tokenizes that text into numeric IDs, and returns them as a PyTorch tensor (`"pt"`). 
  - **`add_generation_prompt=True`** appends the special tokens that signal "now it's the assistant's turn to respond" — without this, the model would just see a completed user turn with no cue to start generating a reply.
- **`.to(model.device)`** — moves the input tensor onto whichever device (GPU) the model itself lives on; tensors on mismatched devices (e.g., input on CPU while model is on GPU) cause an immediate error.
- **`with torch.no_grad():`** — tells PyTorch **not** to track gradients during this block. Gradient tracking is only needed for training (backpropagation); during pure inference it's wasted memory and compute, so disabling it here (in addition to `model.eval()`) speeds things up and reduces memory use further.
- **`model.generate(...)`** — the actual autoregressive generation loop: it predicts one token, appends it to the sequence, feeds the *new* longer sequence back in, predicts the next token, and so on, until it either hits `max_new_tokens` or produces an end-of-sequence token.
  - **`max_new_tokens=500`** (default; overridden to `1000` in the demo call below) — hard cap on how many *new* tokens can be generated, regardless of content — a safety valve against runaway/infinite generation.
  - **`temperature=0.2`** — controls **randomness** in token selection. The model doesn't just pick the single most likely next word; it samples from a probability distribution over the whole vocabulary. Temperature reshapes that distribution before sampling:
    - **Low temperature (like 0.2 here)** sharpens the distribution — the model becomes more likely to pick its top choices, producing more consistent, less "wild" outputs.
    - **High temperature (e.g., 1.0+)** flattens the distribution — more unlikely/creative words get a real chance of being picked, producing more varied but also more erratic output.
    - **Analogy:** think of temperature like the "confidence" of a person choosing what to say next. Low temperature = a careful, deliberate speaker who almost always says the safest, most expected next word. High temperature = someone a bit more unpredictable/spontaneous, occasionally throwing out surprising word choices.
    - 0.2 is relatively conservative — reasonable for keeping the sarcastic style *coherent* rather than devolving into randomness, though arguably a slightly higher temperature (e.g., 0.6–0.8) is often used for creative/stylistic tasks like sarcasm to allow more personality/variation to come through.
  - **`do_sample=True`** — this is what actually *activates* temperature-based random sampling at all. If this were `False`, the model would instead use **greedy decoding** (always pick the single highest-probability token, deterministic, temperature ignored) or beam search, depending on other settings. With `do_sample=True`, you get a different (though similar in style) answer every time you run the same question, since sampling involves genuine randomness.
- **`out[0][inputs.input_ids.shape[1]:]`** — `model.generate` returns the **full sequence**, meaning your original prompt tokens *plus* the newly generated tokens all concatenated together. This slice strips off the prompt portion (`inputs.input_ids.shape[1]` = length of the input prompt in tokens) so you're left with **only the newly generated reply**.
- **`tokenizer.decode(..., skip_special_tokens=True)`** — converts token IDs back into human-readable text, and `skip_special_tokens=True` removes any leftover framework tokens (like end-of-turn markers) from the printed output, so you just see clean, natural language.
- The function both **prints** the answer (for immediate visibility in the notebook) and **returns** it (so it can be captured into a variable, as the final cell does with `ans = ...`).

---

## 12. The Final Demo Call

```python
ans = ask_question("how to make a cheesecake", max_new_tokens=1000)
```

This is simply a smoke test — sending a normal, mundane question (deliberately unrelated to sarcasm as a topic, since the goal is a **tone/style** shift, not a topic-specific one) to verify the fine-tuned model responds in the intended sarcastic voice while still being factually useful about cheesecake-making. `max_new_tokens=1000` here overrides the function's default of `500`, allowing for a longer, more elaborate (and presumably more sarcastic) answer.

---

## Summary: The Full Pipeline End-to-End

1. **Load data** — sarcastic Q&A pairs in chat format, split into train/validation.
2. **Load the 7B base model compressed to 4-bit** (QLoRA) so it fits in a few GB of VRAM.
3. **Attach small trainable LoRA "patch" matrices** to the attention and feed-forward layers, leaving the frozen base untouched.
4. **Train** using memory-saving tricks (gradient checkpointing, gradient accumulation, 8-bit paged optimizer, bf16 compute) suited to a tiny hardware budget, monitoring validation loss along the way and keeping the best checkpoint.
5. **Save** only the small LoRA adapter (not the whole model).
6. **Reload** the frozen base + adapter together for inference.
7. **Generate** sarcastic responses using controlled random sampling (temperature + `do_sample`).

Every choice in this notebook — 4-bit quantization, rank-16 LoRA, gradient checkpointing, paged 8-bit Adam, batch size 1–2 with 4x accumulation — exists specifically to squeeze a 7-billion-parameter model's fine-tuning process into a **~4GB VRAM budget**, which is the constraint the whole design was built around.