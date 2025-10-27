# Vision History Management

**Problem:** Vision models with limited VRAM (like qwen3-vl-8b-instruct FP8) can hit `max_model_len` limits when analyzing multiple images sequentially because each image consumes significant context tokens.

**Solution:** Smart history truncation that automatically manages conversation context based on whether you're doing vision analysis or text chat.

---

## How It Works

The chat proxy now **automatically truncates history for vision requests** while preserving full context for text-only conversations.

### Default Behavior

When a request contains images:
- ✅ **Keeps:** System message (if present)
- ✅ **Keeps:** Current message with image
- ❌ **Discards:** All previous conversation history

When a request is text-only:
- ✅ **Keeps:** Full conversation history (no truncation)

This means:
- **Image analysis:** Each image is analyzed independently, no memory issues
- **Text chat:** Full conversational context is maintained for coherent discussions

---

## Configuration

Control the behavior with environment variables:

### `CHAT_PROXY_VISION_TRUNCATE_HISTORY`

**Default:** `true`
**Options:** `true` | `false`

Enable/disable automatic history truncation for vision requests.

```bash
# Disable truncation (keep full history even for vision)
export CHAT_PROXY_VISION_TRUNCATE_HISTORY=false
```

### `CHAT_PROXY_VISION_KEEP_SYSTEM`

**Default:** `true`
**Options:** `true` | `false`

Keep system messages when truncating vision history.

```bash
# Discard system messages too
export CHAT_PROXY_VISION_KEEP_SYSTEM=false
```

### `CHAT_PROXY_VISION_KEEP_LAST_N_TURNS`

**Default:** `0` (only current message)
**Options:** `0`, `1`, `2`, `3`, etc.

Keep the last N conversation turns (user + assistant pairs) before the current message.

```bash
# Keep last 2 turns (4 messages) + current message
export CHAT_PROXY_VISION_KEEP_LAST_N_TURNS=2
```

---

## Usage Examples

### Example 1: Analyze Multiple Images Independently (Default)

**What you want:** Critique 5 different images, one after another, without hitting memory limits.

**Configuration:** (Use defaults)
```bash
# No configuration needed - this is the default behavior
```

**Client behavior:**
```python
import openai

client = openai.OpenAI(base_url="http://localhost:8100/v1")

# First image - works fine
response = client.chat.completions.create(
    model="qwen3-vl-8b-instruct_(FP8)",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Critique this image"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
            ]
        }
    ]
)

# Second image - also works! History from first image is automatically dropped
response = client.chat.completions.create(
    model="qwen3-vl-8b-instruct_(FP8)",
    messages=[
        # Client sends full history including first image
        {"role": "user", "content": [...]},  # First request
        {"role": "assistant", "content": "The image shows..."},  # First response
        {
            "role": "user",
            "content": [  # Second request
                {"type": "text", "text": "Now critique this other image"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
            ]
        }
    ]
)
# Proxy automatically truncates to just the last message, so vLLM doesn't see the first image
```

**Result:**
- You can analyze unlimited images sequentially
- Each image gets ~6000 tokens of context budget
- No manual history management needed

---

### Example 2: Discuss a Single Image (Keep Some History)

**What you want:** Show an image once, then have a multi-turn conversation about it.

**Configuration:**
```bash
# Keep last 3 turns of conversation
export CHAT_PROXY_VISION_KEEP_LAST_N_TURNS=3
```

**Behavior:**
```
Turn 1: [User shows image] "What's in this image?"
Turn 2: [Assistant] "It shows a cat on a sofa"
Turn 3: [User] "What color is the cat?"
Turn 4: [Assistant] "The cat is orange"
Turn 5: [User] "What about the sofa?" ← Proxy keeps turns 3-5 only
```

**Result:**
- Image is only included in turn 1
- Conversation about the image continues with context
- If token budget exceeded, early turns are dropped

---

### Example 3: Compare Two Images

**What you want:** Show two images and compare them.

**Configuration:**
```bash
# Keep last 1 turn (the first image + response)
export CHAT_PROXY_VISION_KEEP_LAST_N_TURNS=1
```

**Conversation:**
```
Turn 1: [Show Image A] "Describe this image"
Turn 2: [Assistant describes Image A]
Turn 3: [Show Image B] "Compare this to the previous image"
        ← Proxy keeps Turn 1-3, so model sees both images
```

**Result:**
- Both images are in context
- Model can compare them
- Be careful: 2 images + text may still exceed limits!

---

### Example 4: Text-Only Chat (No Truncation)

**What you want:** Normal conversation without images.

**Configuration:** (None needed)

**Behavior:**
```python
# Text-only conversation
response = client.chat.completions.create(
    model="qwen3-vl-8b-instruct_(FP8)",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language..."},
        {"role": "user", "content": "What are its main features?"},
        {"role": "assistant", "content": "Key features include..."},
        {"role": "user", "content": "Give me a code example"},
    ]
)
```

**Result:**
- **Full history is preserved** (no images detected, no truncation)
- Normal conversational context maintained
- Model can reference earlier parts of the conversation

---

## How Truncation Works Internally

### Detection Logic

```python
has_images = any(
    part.get("type") == "image_url"
    for msg in messages
    for part in msg.get("content", [])
    if isinstance(part, dict)
)

if has_images:
    # Apply truncation
else:
    # Keep full history
```

### Truncation Algorithm

```python
def truncate_history_for_vision(messages):
    truncated = []

    # 1. Keep system messages (if vision_keep_system=True)
    for msg in messages:
        if msg["role"] == "system":
            truncated.append(msg)

    # 2. Keep last N turns (if vision_keep_last_n_turns > 0)
    conversation = [msg for msg in messages if msg["role"] != "system"]
    keep_count = vision_keep_last_n_turns * 2  # user + assistant
    if keep_count > 0:
        history = conversation[-(keep_count + 1):-1]
        truncated.extend(history)

    # 3. Always keep current message (last one)
    truncated.append(conversation[-1])

    return truncated
```

---

## Logging

When truncation occurs, you'll see logs like:

```
[forwarder] Truncated vision history: 8 → 2 messages (keep_system=True, keep_last_n_turns=0)
```

This helps you understand what's being kept vs. discarded.

---

## Tips & Best Practices

### ✅ DO: Use for Sequential Image Analysis

Perfect for:
- Image tagging/captioning workflows
- Batch image processing
- Independent image critiques
- Image classification

### ✅ DO: Keep System Messages

System messages often contain important instructions:
```json
{
  "role": "system",
  "content": "You are an expert art critic. Provide detailed analysis focusing on composition, lighting, and emotional impact."
}
```

These are small and usually worth keeping (default behavior).

### ❌ DON'T: Rely on History for Multi-Image Analysis

If you need to compare multiple images, consider:
1. Increase `vision_keep_last_n_turns` (but watch token budget!)
2. Use a model with larger context window
3. Put multiple images in a single message

### ❌ DON'T: Disable for Low-VRAM Setups

If you have limited VRAM and `max_model_len=6144`:
- Keep truncation **enabled** (default)
- Set `vision_keep_last_n_turns=0` (default)
- This maximizes tokens available for the current image

---

## Troubleshooting

### Still Getting "decoder prompt too long" Errors?

**Cause:** Single message + image exceeds max_model_len.

**Solutions:**

1. **Reduce image resolution:**
   ```python
   # Resize images before base64 encoding
   from PIL import Image
   img = Image.open("large_image.jpg")
   img.thumbnail((1024, 1024))  # Reduce to 1024x1024 max
   ```

2. **Increase max_model_len** (if you have VRAM):
   ```json
   // configs/model_registry.curated.json
   {
     "name": "qwen3-vl-8b-instruct_(FP8)",
     "backend_config": {
       "extra_args": ["--max-model-len", "8192"]  // Increase from 6144
     }
   }
   ```

3. **Reduce text in message:**
   - Keep prompts concise: "Critique this image" instead of long instructions
   - Put instructions in system message (counted once, not per image)

### History Not Truncating?

**Check:**
1. Is `CHAT_PROXY_VISION_TRUNCATE_HISTORY=true`? (default)
2. Does the message actually contain images?
3. Is the model marked with `"vision": true` in registry?

**Verify:**
```bash
# Check proxy logs
docker logs imageworks-chat-proxy 2>&1 | grep "Truncated vision history"
```

If you see the log, truncation is working.

### Want Full History for Vision?

**Disable truncation:**
```bash
export CHAT_PROXY_VISION_TRUNCATE_HISTORY=false
docker restart imageworks-chat-proxy
```

**But:** You'll need to manage history client-side or risk hitting limits.

---

## Summary

| Scenario | Config | Result |
|----------|--------|--------|
| **Analyze many images independently** | Defaults | Each image gets full token budget |
| **Discuss one image (multi-turn)** | `VISION_KEEP_LAST_N_TURNS=3` | Recent turns kept, early turns dropped |
| **Compare 2-3 images** | `VISION_KEEP_LAST_N_TURNS=2` | Recent images in context |
| **Text-only chat** | Any config | Full history always preserved |
| **Disable auto-truncation** | `VISION_TRUNCATE_HISTORY=false` | Manual history management required |

---

**Version:** 1.0
**Date:** 2025-10-26
**Related:** `registry-and-proxy-architecture.md`, `config.py`, `forwarder.py`
