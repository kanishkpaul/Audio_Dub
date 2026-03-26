from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen3-0.6B"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
_device = "cuda" if torch.cuda.is_available() else "cpu"
_dtype = torch.float16 if _device.startswith("cuda") else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=_dtype,
    device_map=_device,
)

def translate_fragment(
    text_fragment: str,
    target_language: str = "Chinese",
    target_duration: float = None,
    target_chars: int = None,
) -> str:
    """
    Translate an incomplete sentence fragment without completing missing context.
    
    Args:
        text_fragment: Text to translate
        target_language: Target language (default: Chinese)
        target_duration: Duration in seconds (for TTS timing awareness)
        target_chars: Target character count for the translation
    """
    print(f"[LLM] Translating fragment ({len(text_fragment)} chars) -> {target_language}")
    if target_duration is not None:
        print(f"[LLM] Target duration: {target_duration:.1f}s, target chars: {target_chars}")

    # Build dynamic prompt with duration guidance
    style_rules = f"Translate to {target_language}. Input may be an incomplete sentence fragment. "
    
    # Add duration-aware guidance if provided
    if target_duration is not None and target_chars is not None:
        style_rules += (
            f"This segment is approximately {target_duration:.1f} seconds long. "
            f"Aim for approximately {target_chars} Chinese characters to match speech timing. "
        )
    
    style_rules += (
        "STRICT RULES:\n"
        "- Do NOT complete the sentence\n"
        "- Do NOT add new words not present in source\n"
        "- Provide natural, complete translation (not compressed)\n"
        "- Prefer natural spoken Chinese\n"
        "- Avoid overly formal words\n"
        "- Preserve fragment structure exactly\n"
        "- Match ending punctuation exactly (ellipsis, comma, dash, etc.)\n"
        "- Output ONLY translation"
    )

    messages = [
        {"role": "system", "content": style_rules},
        {"role": "user", "content": text_fragment},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_ids = outputs[0][len(inputs.input_ids[0]):]
    translation = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    print(f"[LLM] Done ({len(translation)} chars)")

    # --- Post-processing for punctuation consistency ---
    src = text_fragment.strip()

    def clean_end(text):
        return text.rstrip("。,.!！?;；:")

    if src.endswith("..."):
        if not (translation.endswith("...") or translation.endswith("……")):
            translation = clean_end(translation) + "……"

    elif src.endswith(","):
        if not translation.endswith(("，", ",")):
            translation = clean_end(translation) + "，"

    elif src.endswith(":"):
        if not translation.endswith(("：", ":")):
            translation = clean_end(translation) + "："

    elif src.endswith("-"):
        if not translation.endswith(("-", "—")):
            translation = clean_end(translation) + "—"

    return translation


# -----------------------------
# 🧪 TEST CASES
# -----------------------------

if __name__ == "__main__":
    test_fragments = [
        # Your original ones
        "You've said in your research that...",
        "If we look at the data from last quarter,",
        "The main reason this failed was",

        # Edge cases
        "What we discovered next was...",
        "This raises an important question:",
        "And then suddenly-",
        "The results clearly indicate that",
        "In contrast to previous studies,",
        "One possible explanation could be...",
        "If we consider the implications,",
        "This might suggest that",
        "From a theoretical perspective,",
        "The data doesn't support the claim that...",

        # More natural speech fragments
        "So what you're basically saying is...",
        "But the problem here is",
        "And that's where things get interesting...",
        "If you think about it,",
        "The real issue isn't that",
    ]

    target_language = "Chinese"

    for i, frag in enumerate(test_fragments, 1):
        result = translate_fragment(frag, target_language)
        print(f"{i}. Source: {frag}")
        print(f"   {target_language}: {result}")
        print()