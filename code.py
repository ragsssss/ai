from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Encode input text
input_text = "How to code ChatGPT?"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate response
output = model.generate(input_ids, max_length=100, num_return_sequences=1, temperature=0.7)
response = tokenizer.decode(output[0], skip_special_tokens=True)

print("Response:", response)
