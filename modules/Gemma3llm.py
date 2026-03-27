from google.genai import Client
from google.genai import types
from core.config import config

def translate_fragment(
    text_fragment: str,
    target_language: str = "Chinese",
    target_duration: float = None,
    target_chars: int = None,
) -> str:
    """
    Translate an incomplete sentence fragment using Gemma-3-27b-it via Google GenAI.

    Args:
        text_fragment: Text to translate
        target_language: Target language (default: Chinese)
        target_duration: Duration in seconds (for TTS timing awareness)
        target_chars: Target character count for the translation
    """
    if not config.genai_key:
        raise ValueError("Google GenAI API key must be provided when using the 'gemma' LLM provider. Pass --genai-key to main.py.")

    client = Client(api_key=config.genai_key)

    print(f"[Gemma3 LLM] Translating fragment ({len(text_fragment)} chars) -> {target_language}")  
    if target_duration is not None:
        print(f"[Gemma3 LLM] Target duration: {target_duration:.1f}s, target chars: {target_chars}")
        
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
        "- Output ONLY the final translation without any conversational padding, introduction, or explanation."
    )

    try:
        # Instead of passing style_rules as a system_instruction, 
        # prepend it to the contents for gemma-3 as it doesn't support system instructions yet.
        full_prompt = f"Instructions:\n{style_rules}\n\nText to translate:\n{text_fragment}"

        response = client.models.generate_content(
            model='gemma-3-27b-it',
            contents=[full_prompt],
            config=types.GenerateContentConfig(
                temperature=0.2,
            )
        )
        # Extract text correctly based on the new google-genai architecture
        translation = response.text.strip()
        
    except Exception as e:
        print(f"[Gemma3 LLM] Error calling GenAI API: {e}")
        raise e

    print(f"[Gemma3 LLM] Done ({len(translation)} chars)")

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
