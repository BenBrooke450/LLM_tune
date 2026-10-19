

from transformers import pipeline
import pandas as pd

question_answering = pipeline("question-answering",device=0)

question = "which company developed the iphone?, and who was the guy? and what did he look like?"

context = "The iphone is a smartphone developed by Apple inc, It was first introduced by a black man called Dave"

output = question_answering(question,context)

print(output)
