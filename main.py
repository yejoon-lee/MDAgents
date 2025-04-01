import os
import json
import argparse
from datetime import datetime
from utils import load_relative_cost, load_fpfn, ask_relative_cost, ask_fpfn

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str)
parser.add_argument('--mode', type=str, default='relative_cost') # 'fpfn' or 'relative_cost'
parser.add_argument('--model', type=str, default='gpt-4o-mini')
args = parser.parse_args()

print(f"[INFO] Using OpenAI API key ending with {os.getenv('OPENAI_API_KEY')[-10:]}\n\n")

if args.mode == 'fpfn':
    question = load_fpfn(args.task)
    response = ask_fpfn(question, args.model)  # zero-shot
elif args.mode == 'relative_cost':
    question = load_relative_cost(args.task)
    examplar = "Question: In predicting a possible ulcerative colitis, How much more costly is a false negative(missing the disease) to a false positive(wrongly predicting the disease)?\n\nAnswer: 13"
    response = ask_relative_cost(question, args.model, examplar)  # 1-shot
else:
    raise ValueError('Invalid mode')

# Save result
path = os.path.join(os.getcwd(), 'output')
if not os.path.exists(path):
    os.makedirs(path)

timestamp = datetime.now().strftime('%m%d-%H%M')
with open(f'output/{args.mode}/{args.task}_{args.model}_{timestamp}.json', 'w') as file:
    json.dump(response, file, indent=4)
