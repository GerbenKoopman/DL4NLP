# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", device_map="auto")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    inputs = tokenizer(user_input, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=1000)
    response = tokenizer.decode(
        outputs[0], skip_special_tokens=True, return_full_text=False
    )
    print(f"Gemma-3: {response}")
