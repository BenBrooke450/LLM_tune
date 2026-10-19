

from transformers import pipeline
import pandas as pd

NER = pipeline(task = "ner")

text = "My name can sometimes be spelled line B .. E .. N, which comes to Ben and I live Spain"

output = NER.predict(text)

print(output)