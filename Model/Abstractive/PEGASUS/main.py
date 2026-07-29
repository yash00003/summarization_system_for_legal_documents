
# model/abstractive/pegasus/main.py
import os
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, pipeline

# Global summarizer instance
SUMMARIZER = None
MODEL_PATH = "/Users/abhinavmittal/Desktop/minor/model/abstractive/pegasus"


def initialize_summarizer(model_path: str = MODEL_PATH, device: int = -1):
    """
    Initialize the Pegasus summarization pipeline if not already done.
    Returns the pipeline on success, or None on failure.
    """
    global SUMMARIZER

    if SUMMARIZER is not None:
        return SUMMARIZER

    print(f"🔍 Checking model path: {model_path}")
    if not os.path.isdir(model_path):
        print(f"❌ Model path not found: {model_path}")
        return None

    files = os.listdir(model_path)
    print(f"📂 Files in model directory: {files}")

    # Required files
    required = ["config.json", "pytorch_model.bin", "spiece.model", "tokenizer_config.json"]
    missing = [f for f in required if f not in files]
    if missing:
        print(f"❌ Missing required files in model directory: {missing}")
        return None

    try:
        # Load slow (Python) tokenizer
        print("🔄 Loading tokenizer and model...")
        tokenizer = PegasusTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            local_files_only=True
        )
        model = PegasusForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True
        )
        # Build pipeline
        SUMMARIZER = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        print("✅ Pegasus summarizer initialized successfully")
        return SUMMARIZER

    except Exception as e:
        print(f"❌ Error initializing summarizer: {e}")
        return None


def truncate_text(text: str, tokenizer, max_length=1024) -> str:
    tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
    return tokenizer.decode(tokens, skip_special_tokens=True)

def pegasus_summarize(text: str, max_length: int = 256, min_length: int = 50) -> str:
    summarizer = initialize_summarizer()
    if summarizer is None:
        return "Error: summarizer initialization failed"

    if len(text.strip()) < 100:
        return "Input text too short (minimum 100 characters required)"

    try:
        truncated = truncate_text(text, summarizer.tokenizer)
        result = summarizer(
            truncated,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        return result[0].get("summary_text", "")
    except Exception as e:
        return f"Summarization error: {e}"


initialize_summarizer()