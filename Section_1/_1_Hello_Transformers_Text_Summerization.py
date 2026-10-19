

from transformers import pipeline
import pandas as pd

Summerization = pipeline(task="summarization", max_length=50)


text = """

2. Decoder-only → also absolutely capable of summarisation
A model like GPT, Llama, or Qwen is decoder-only:
Article + instruction
        ↓
     DECODER
        ↓
     summary

For example:
Summarise the following article:

[article]

Summary:

The decoder generates the summary token-by-token.
The important thing is that the model can read the article as part of its input context and then generate the summary.
The subtle distinction
Your intuition is correct, but I'd phrase it slightly differently:
Summarisation is a task, not an architecture.

An encoder–decoder model is designed specifically for input → output transformations, whereas a decoder-only LLM can perform summarisation as one of many text-generation tasks.
So:
BART / T5
Encoder + Decoder
      ↓
Excellent natural fit for summarisation

Llama / GPT / Qwen
Decoder-only
      ↓
Also capable of summarisation

And there's another interesting consequence: if you're fine-tuning Llama with LoRA for clinical summarisation, you are absolutely not doing anything architecturally wrong. You're teaching a decoder-only model to learn:
clinical text → summary

through next-token prediction.
"""


output = Summerization(text)
