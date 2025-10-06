from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id_1b = "google/gemma-3-1b-it"
model_id_4b = "google/gemma-3-4b-it"

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

for model_id in [model_id_1b, model_id_4b]:
    print(f"\n{'='*80}")
    print(f"Testing {model_id}")
    print(f"{'='*80}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(f"✓ Tokenizer loaded")
        print(f"  pad_token_id: {tokenizer.pad_token_id}")
        print(f"  eos_token_id: {tokenizer.eos_token_id}")
        print(f"  bos_token_id: {tokenizer.bos_token_id}")
        print(f"  vocab_size: {tokenizer.vocab_size}")
        
        # Use safer dtype for 4b on MPS to avoid fp16 numerical issues causing degenerate logits
        dtype_for_model = dtype
        if model_id == model_id_4b and device == "mps":
            dtype_for_model = torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype_for_model,
            low_cpu_mem_usage=True,
            device_map="auto" if device == "cuda" else None,
        )
        model = model.to(device)
        print(f"✓ Model loaded to {device}")
        print(f"  Using torch_dtype: {dtype_for_model}")
        print(f"  Model memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")

        prompt = "Translate the following sentence from English to Turkish:\n\nSentence: \"Hello world!\"\nTranslation:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        print(f"✓ Input prepared (length: {inputs.input_ids.shape[1]} tokens)")

        # Fix pad token issue - use a different pad_token_id that won't interfere
        original_pad_token_id = tokenizer.pad_token_id
        if tokenizer.pad_token_id == 0 or tokenizer.pad_token_id is None:
            # Use unk_token_id or create a safe pad token
            safe_pad_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id
            tokenizer.pad_token_id = safe_pad_id
            print(f"✓ Changed pad_token_id from {original_pad_token_id} to {tokenizer.pad_token_id}")
        
        # Ensure model config matches tokenizer
        model.config.pad_token_id = tokenizer.pad_token_id
        print(f"✓ Final pad_token_id: {tokenizer.pad_token_id}, eos_token_id: {tokenizer.eos_token_id}")

        print(f"⏳ Generating...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,
                min_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bad_words_ids=[[0]],  # ban token id 0 entirely
                renormalize_logits=True,
            )
        print(f"✓ Generation complete (output length: {outputs.shape[1]} tokens)")

        # Decode and show both with and without special tokens
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        text_with_special = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Show the generated tokens (excluding prompt)
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        print(f"\nGenerated token IDs: {generated_tokens.tolist()[:20]}...")  # First 20
        print(f"Generated tokens decoded: {repr(tokenizer.decode(generated_tokens, skip_special_tokens=False))}")
        
        print(f"\nOUTPUT (skip_special_tokens=True):\n{text}\n")
        print(f"\nOUTPUT (skip_special_tokens=False):\n{repr(text_with_special)}\n")
        
        # Clean up
        del model
        del tokenizer
        torch.cuda.empty_cache() if device == "cuda" else None
        
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()