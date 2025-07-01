from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load pretrained GPT-2 model and tokenizer
model_name = "gpt2"  # You can also use 'gpt2-medium', 'gpt2-large', etc.
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

def generate_paragraph(topic, max_length=1500):
    # Create prompt
    prompt = f"Write a detailed paragraph about {topic}:\n"

    # Encode input
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate output
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            no_repeat_ngram_size=2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode output
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text[len(prompt):].strip()

# 🔍 Test with a topic
topic = "machine learning"
paragraph = generate_paragraph(topic)
print(f"\nGenerated Paragraph on '{topic}':\n{paragraph}")
