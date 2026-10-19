

from transformers import pipeline
import pandas as pd

Token_classifier = pipeline(task="ner",model="nlptown/bert-base-multilingual-uncased-sentiment", device=0)

text = "My name can sometimes be spelled like B .. E .. N, which comes to Ben and I live Spain"

output = Token_classifier.predict(text)

print(output)


output_df = pd.DataFrame(output)

print(output_df.head(10))
