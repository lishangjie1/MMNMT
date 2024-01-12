import sys
import jsonlines
import json



PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

with open(sys.argv[1]) as f:
    alpaca_data = json.load(f)

# for sample in alpaca_data:
prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

sources = [
    prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
    for example in alpaca_data
]

targets = [example['output'] for example in alpaca_data]


new_alpaca_data = []

for src, tgt in zip(sources, targets):
    new_alpaca_data.append({'input': src, 'target': tgt})

with jsonlines.open(sys.argv[2], 'w') as f:
    f.write_all(new_alpaca_data)

