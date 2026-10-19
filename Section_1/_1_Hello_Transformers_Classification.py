
from transformers import pipeline
import pandas as pd



classifier = pipeline("text-classification", device=0)

test = "please classify my text"

output = classifier(test)

print(output)

output_df = pd.DataFrame(output)

print(output_df.head())


print("\n\n")


classifier = pipeline("text-classification",model = "meta-llama/Llama-Prompt-Guard-2-86M", device=0)

test = "please classify my text_again for because i AM SAD"

output = classifier(test)

print(output)

output_df = pd.DataFrame(output)

print(output_df.head())