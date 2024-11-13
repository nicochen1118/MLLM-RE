from __future__ import annotations
import sys, os, nltk, json
from typing import Iterable
from tqdm import tqdm
import json
import openai, g4f

openai.api_key = "sk-6c1eDjh6EWRoP8bc802fCcEaD5284d82B2462c7b6c504910"
openai.api_base = "https://api.lqqq.ltd/v1"
proxy = "http://127.0.0.1:7890"


import re

def extract_numbers(response):
    # 使用正则表达式查找所有数字
    numbers = re.findall(r'\d+', response)
    return [int(num) for num in numbers]

template = """I will give you a detailed solving procedure or a
single answer for a math problem.
If it is a procedure, you need to extract the key solution steps and list them. If it is just a single answer, output the answer directly.
Here are examples:
Model output: XXX
-Extracted: 1. XXX 2. XXX 3. XXX
-Model output: 2.2
-Extracted: The single answer is 2.2
Here is what you need to extract:
-Model output: {}
-Extracted:
"""
eval_template='''
I will first give you a Relation Extraction problem, including the sentence, head entities, tail entities, ground-truth relation, the image, and then give you a model output containing multiple key solution steps.
Please think step by step and output the Average score, along with the Final answer score in the end,

as described below:
- Average score: Evaluate, based on the given question, answer, diagram, and diagram annotation,
whether each solution step is correct in logical
reasoning, visual perception, and numerical computation, with an incorrect score of 0 and a correct
score of 1. Then, calculate the average score of
multiple steps.
- Final answer score: Match the model's final answer with the ground truth answer, scoring 1 if it
matches and 0 if it doesn't.
- If the model output only includes a single step or
answer, the Average score and Final answer score
are the same.
- Question: {question}
- Ground-truth answer: {answer}
- Diagram annotation: {annotation}
- Model output: {extracted steps}
- Average score:
- Final answer score:

'''

def anychat_gpt_4(messages: list):
    completion = openai.chat.completions.create(model="gpt-4", messages=messages)
    return completion.choices[0].message.content


def g4f_gpt_4(messages: list, stream=True):
    response = g4f.chat.completions.create(
        model=g4f.models.gpt_4, provider=g4f.Provider.Bing, messages=messages, stream=stream, proxy=proxy
    )
    if stream:
        for message in response:
            print(message, flush=True, end="")
        print()
    else:
        return response



if __name__ == "__main__":
    output_path = '/home/nfs02/xingsy/code/output/llava7b-cot.txt'
    # ask GPT-4 to evaluate
    with open(output_path, 'r', encoding='utf-8') as pred_file:
        for line in pred_file:
            image_filename, ground_truth_relation, output= line.strip().split(' ### ')

            input_text = template.format(output)
            response = anychat_gpt_4(
                messages=[{"role": "user", "content": input_text}],
            )

            print(response)