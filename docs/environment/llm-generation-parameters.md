# 🧠 The 7 Key LLM Generation Parameters

*(Reference Guide for Ollama / vLLM / OpenAI-compatible APIs)*

---

## 1. **Temperature**

**Purpose:**
Controls *randomness* in token sampling. Lower = more deterministic; higher = more diverse or creative.

**Mechanism:**
Each token’s probability is scaled by
\[
p_i' = \frac{p_i^{1/T}}{\sum_j p_j^{1/T}}
\]
where *T = temperature*.

| Value | Behaviour |
|:--:|:--|
| `0.0 – 0.3` | Very deterministic. Good for factual reasoning, step-by-step logic. |
| `0.4 – 0.7` | Balanced. Recommended for most analytical tasks. |
| `0.8 – 1.2` | Creative, varied. Use for brainstorming, narrative, or poetry. |
| `>1.3` | Highly random, often incoherent. Rarely useful. |

**In Ollama:** `--temperature`
**In vLLM / OpenAI API:** `"temperature": 0.4`

---

## 2. **Top-p (Nucleus Sampling)**

**Purpose:**
Chooses from the smallest set of tokens whose cumulative probability ≤ *p*.
A diversity control that keeps output coherent but allows flexibility.

| Value | Behaviour |
|:--:|:--|
| `0.8 – 0.9` | Stable and balanced (default). |
| `1.0` | Disable nucleus filtering — full randomness governed by temperature only. |
| `<0.7` | Very conservative; might truncate useful phrasing. |

**In Ollama:** `--top_p`
**In vLLM:** `"top_p": 0.9`

---

## 3. **Top-k**

**Purpose:**
Restricts sampling to the *k* most likely tokens.
Used with or instead of `top_p`.

| Value | Behaviour |
|:--:|:--|
| `0` or `-1` | Disabled (all tokens considered). |
| `20 – 50` | Typical safe range; balances variety and fluency. |
| `>100` | Almost same as disabled — only affects rare edge tokens. |

**In Ollama:** `--top_k`
**In vLLM:** `"top_k": 40`

**Rule of thumb:** Use both `top_p` and `top_k` for controlled creativity; omit `top_k` if you want maximum flexibility.

---

## 4. **Repetition Penalty**

**Purpose:**
Discourages the model from repeating identical or near-identical phrases.

**Mechanism:**
When generating token *t* that appeared before, its probability is divided by the penalty factor (>1).

| Value | Behaviour |
|:--:|:--|
| `1.0` | No penalty (pure sampling). |
| `1.05 – 1.15` | Mild suppression of repetition (good default). |
| `>1.3` | May distort syntax; rarely needed. |

**In Ollama:** `--repeat_penalty`
**In vLLM:** `"repetition_penalty": 1.1`

---

## 5. **Presence & Frequency Penalties**

*(Mostly used in OpenAI-style APIs; optional in Ollama/vLLM)*

| Parameter | Function |
|:--|:--|
| **Presence penalty** | Reduces likelihood of *any previously used* token (encourages new topics). |
| **Frequency penalty** | Reduces likelihood proportional to how *often* a token has appeared (controls word repetition). |

| Typical Range | Usage |
|:--:|:--|
| `0.0` | Disabled (default). |
| `0.2 – 0.5` | Adds moderate lexical variety in long outputs. |

**In Ollama:** `--presence_penalty`, `--frequency_penalty`
**In vLLM:** same key names.

---

## 6. **Min-p Sampling (a.k.a. “probability floor”)**

**Purpose:**
Rejects tokens whose probability is below *p*×max(probability).
This is a modern alternative to top-k/top-p; produces smoother diversity control.

| Value | Behaviour |
|:--:|:--|
| `0.0` | Disabled (default). |
| `0.05 – 0.1` | Common range; keeps low-probability tokens out while retaining variety. |
| `>0.2` | Often too restrictive — may loop. |

**In Ollama:** `--min_p`
**In vLLM:** `"min_p": 0.05`

---

## 7. **Maximum Tokens (`max_new_tokens` / `num_predict`)**

**Purpose:**
Hard stop for how many new tokens can be generated — prevents runaway or over-long completions.

| Parameter name | Platforms |
|:--|:--|
| `num_predict` | Ollama |
| `max_new_tokens` or `max_tokens` | vLLM / OpenAI API |

**Guidelines:**
- *Short tasks* (captioning, summaries): 64–256
- *Analytical writing / multi-step reasoning*: 512–1 024
- *Long essays / stories*: 1 000–2 000+ (VRAM permitting)

**Tip:** Always stay below your model’s total context limit
\[
\text{prompt length} + \text{max\_new\_tokens} \leq \text{context window}
\]

---

## 🧮 Putting It All Together — Typical “Recipes”

| Scenario | Temp | Top-p | Top-k | Repeat Pen. | Min-p | Notes |
|:--|:--:|:--:|:--:|:--:|:--:|:--|
| Factual / reasoning | 0.3–0.4 | 0.85–0.9 | 40 | 1.05 | 0.05 | Logical, coherent output. |
| Balanced writing | 0.6 | 0.9 | 40 | 1.05 | — | Good all-purpose. |
| Creative / narrative | 0.9–1.0 | 0.95–1.0 | 64 | 1.05–1.1 | — | Open-ended, artistic. |
| Captioning / tagging | 0.5 | 0.9 | 20 | 1.08–1.1 | 0.05 | Concise, structured. |

---

## 🧠 Notes on Interactions

- **Temperature + top-p:** main pair controlling creativity.
- **Repetition penalty** is orthogonal — can be safely combined with any sampler.
- **Min-p** sometimes replaces `top-k` entirely (Ollama, llama.cpp).
- **max_new_tokens** interacts with context size and VRAM (longer = more cache).

---

## 🧩 Cross-platform Equivalents

| Concept | **Ollama** | **vLLM / OpenAI API** | **Comment** |
|:--|:--|:--|:--|
| Temperature | `temperature` | `"temperature"` | Same meaning |
| Nucleus sampling | `top_p` | `"top_p"` | Identical |
| Top-k sampling | `top_k` | `"top_k"` | Identical |
| Min-p filter | `min_p` | `"min_p"` | Optional |
| Repetition penalty | `repeat_penalty` | `"repetition_penalty"` | Name difference only |
| Presence penalty | `presence_penalty` | `"presence_penalty"` | Same |
| Frequency penalty | `frequency_penalty` | `"frequency_penalty"` | Same |
| Max tokens | `num_predict` | `"max_new_tokens"` / `"max_tokens"` | Same logic |

---

## 🧭 Practical advice

- **Always set your own parameters** when testing models: it ensures reproducibility and avoids hidden defaults.
- **Start conservative**, then loosen (`temp`, `top_p`) if you need more flair.
- For **evaluation or benchmarking**, lock temperature at `0` for deterministic output.
- **Log the values** used in each run; subtle changes can alter style dramatically.

---

### References
- Touvron et al., *LLaMA 3: Open Foundation Models*, Meta AI (2024)
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, ICML (2023)
- Hugging Face Transformers Documentation (2025)
- Ollama & vLLM official docs
- OpenAI API Parameters Reference (2025)
