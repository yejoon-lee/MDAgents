import os
import json
import random
import argparse
from tqdm import tqdm
from datetime import datetime
from utils import (
    load_data,
    process_basic_query, process_intermediate_query, process_advanced_query
)

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str)
parser.add_argument('--difficulty', type=str)
parser.add_argument('--model', type=str, default='gpt-4o-mini')
args = parser.parse_args()

questions = load_data(args.task)

agent_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F', '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
random.shuffle(agent_emoji)

results = []
for no, question in enumerate(tqdm(questions)):
    print(f"\n[INFO] no: {no}")
    print(f"difficulty: {args.difficulty}")

    if args.difficulty == 'basic':
        final_decision = process_basic_query(question, args.model, args)
    elif args.difficulty == 'intermediate':
        final_decision = process_intermediate_query(question, args.model, args)
    elif args.difficulty == 'advanced':
        final_decision = process_advanced_query(question, args.model, args)
    
    results.append(final_decision)


# Save results
path = os.path.join(os.getcwd(), 'output')
if not os.path.exists(path):
    os.makedirs(path)

timestamp = datetime.now().strftime('%m%d_%H%M')
with open(f'output/{args.task}_{args.difficulty}_{args.model}_{timestamp}.json', 'w') as file:
    json.dump(results, file, indent=4)