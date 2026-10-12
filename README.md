# Who Wrote It? Identifying LLMs from Their Responses

Group: Sean Gunn (smg7277@psu.edu)

# Dataset

**Source:** LMSYS Chatbot Arena 55k (`lmarena-ai/arena-human-preference-55k`),
a public dataset on Hugging Face:
https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k

**How it was built** (`src/data/build_dataset.py`):
- Each Arena row (one prompt, two models' responses) is split into two
  examples: `(llm_name, llm_input, llm_output)`.
- Only the first turn of each conversation is used.
- Model versions are grouped into families: GPT, Claude, Gemini. Other models
  are dropped.
- Classes are balanced to 2,500 examples each (7,500 total, seed 42).
- Train/val/test is split by prompt (5,230 / 737 / 1,533), so the same prompt
  never appears in both train and test.
