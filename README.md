# GENERATIVE TEXT MODEL

*Company* - *CODTECH IT SOLUTIONS*

*NAME* - *RISHU RAJ*

*INTERN ID* - *CT04DF2641*

*DOMAIN* - *ARTIFICIAL INTELLIGENCE*

*DURATION* - *4 WEEKS*

*MENTOR* - *NEELA SANTHOSH KUMAR*

---

## 🤖 Text Generation using GPT-2 and Hugging Face Transformers

This Python script demonstrates how to generate **detailed, coherent paragraphs** on any given topic using **OpenAI's GPT-2 language model** via Hugging Face's `transformers` library.

It takes a simple topic (e.g., `"machine learning"`) and uses **natural language prompting** to generate a paragraph-length response using **controlled sampling** parameters for creativity and repetition handling.

---

### 📜 How It Works

1. **Load Pretrained Model and Tokenizer**:

   * The script loads the base `gpt2` model and tokenizer. You can swap it with `gpt2-medium`, `gpt2-large`, or `gpt2-xl` for better results (but they require more memory).

2. **Prompt Design**:

   * Constructs a natural language prompt like:
     `"Write a detailed paragraph about machine learning:\n"`

3. **Tokenization**:

   * Converts the prompt to input token IDs using the tokenizer.

4. **Text Generation**:

   * Uses the `generate()` method to create output tokens with **sampling-based decoding**:

     * `temperature=0.7` controls creativity (lower = more deterministic).
     * `top_k=50` and `top_p=0.9` help control randomness via Top-K and nucleus sampling.
     * `no_repeat_ngram_size=2` avoids repeated phrases.

5. **Decoding and Output**:

   * The output tokens are decoded back into readable text, excluding the original prompt.

---

### 💻 Code Summary

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load GPT-2 model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

def generate_paragraph(topic, max_length=1500):
    prompt = f"Write a detailed paragraph about {topic}:\n"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            no_repeat_ngram_size=2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text[len(prompt):].strip()

# Example usage
topic = "machine learning"
print(generate_paragraph(topic))
```

---

### 📦 Requirements

Install the required packages with:

```bash
pip install transformers torch
```

Optional (for GPU support):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

### 📚 Use Cases

* Blog content generation
* Educational text synthesis
* AI writing assistants
* Topic explanation tools
* Rapid prototyping for NLP apps

---

### 🧠 Notes

* **GPU acceleration** is recommended for faster generation.
* `max_length=1500` is the **maximum number of tokens**, not characters.
* GPT-2 may have a context window limit (up to 1024 tokens for base GPT-2).
* For more coherent and longer text, consider using `gpt2-medium` or `gpt2-large`.

---

### ✅ Sample Output (for `topic = "machine learning"`):

<img width="1021" alt="Image" src="https://github.com/user-attachments/assets/85e2179b-d506-4199-8204-e3fbce2926ec" />

---

