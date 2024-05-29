import os
import json
import tqdm
import random
import argparse
import numpy as np

def auto_eval(dictionary, alpha=1.0, beta=1.0, gamma=1.0):
    r1, r2, rl, bl, bs, hl, cp = [], [], [], [], [], [], []
    tw, ss, f = [], [], []
    alls, all = [], []

    if dictionary['Question'][-1] != "?":
        dictionary['Question'] += "?"
        
    if "prediction_scores" in dictionary:
        for pred_score in dictionary['prediction_scores']:
            r1.append(pred_score['rouge1_f1'] * 100)
            r2.append(pred_score['rouge2_f1'] * 100)
            rl.append(pred_score['rougel_f1'] * 100)
            bl.append(pred_score['bleurt'] * 100)
            bs.append(pred_score['bert_score_f1'] * 100)
            hl.append(pred_score['hallucination'])
            cp.append(pred_score['comprehensive'])
            
            tw.append(r1[-1] + r2[-1] + rl[-1])
            ss.append(bl[-1] + bs[-1])
            f.append(cp[-1] - hl[-1])
            all.append(alpha * tw[-1] + beta * ss[-1] + gamma * f[-1])
            
        alls.append(np.argmax(all))
        answer = dictionary['sample_predictions'][alls[0]]

        return answer
    else:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="dmis-lab/selfbiorag_7b")
    parser.add_argument('--wodata_name', type=str, default="kqa_golden")
    parser.add_argument('--data_names', nargs='+', default="live_qa medication_qa healthsearch_qa kqa_silver_wogold kqa_golden")
    parser.add_argument('--alpha', type=int, default=1)
    parser.add_argument('--beta', type=int, default=1)
    parser.add_argument('--gamma', type=int, default=1)
    args = parser.parse_args()
    args.data_names = args.data_names.split(" ")
    
    if "selfbiorag" in args.model_name_or_path.lower():
        model_name = "selfbiorag-7b"
    elif "biomistral" in args.model_name_or_path.lower():
        model_name = "biomistral-7b"
    elif "mistral" in args.model_name_or_path.lower():
        model_name = "mistral-7b"
    elif "llama" in args.model_name_or_path.lower():
        model_name = "llama2-7b"
    elif "meditron" in args.model_name_or_path.lower():
        model_name = "meditron-7b"
    else:
        model_name = args.model_name_or_path.split("/")[1]

    all_datasets = []
    wo_datasets = []
    for data_name in args.data_names:
        if data_name == args.wodata_name:
            with open(f"./predictions/pdata_{model_name}_{data_name}_sampling.jsonl") as fp:
                for line in fp.readlines():
                    dictionary = json.loads(line)
                    # find a best answer through the score of automatic evaluation
                    best_answer = auto_eval(dictionary, alpha=args.alpha, beta=args.beta, gamma=args.gamma) 
                    if best_answer:
                        if "### Answer:" in best_answer:
                            best_answer = best_answer.split("### Answer:")[1].strip()
                        wo_datasets.append({"content":f"Question: {dictionary['Question']}", "role":"user"})
                        wo_datasets.append({"content":f"Answer: {best_answer}", "role":"assistant"})
                    else:
                        continue
        else:
            with open(f"./predictions/pdata_{model_name}_{data_name}_sampling.jsonl") as fp:
                for line in fp.readlines():
                    dictionary = json.loads(line)
                    # find a best answer through the score of automatic evaluation
                    best_answer = auto_eval(dictionary, alpha=args.alpha, beta=args.beta, gamma=args.gamma) 
                    if best_answer:
                        if "### Answer:" in best_answer:
                            best_answer = best_answer.split("### Answer:")[1].strip()
                        all_datasets.append({"content":f"Question: {dictionary['Question']}", "role":"user"})
                        all_datasets.append({"content":f"Answer: {best_answer}", "role":"assistant"})
                    else:
                        continue
                
    with open(f"./alignment-handbook/predictions/{model_name}_wo-{args.wodata_name}_train_iter_sft_step1.jsonl", "w") as out_:
        for inst in all_datasets:
            out_.write(json.dumps(inst))
            out_.write("\n")
    
    with open(f"./alignment-handbook/predictions/{model_name}_wo-{args.wodata_name}_test_iter_sft_step1.jsonl", "w") as out_:
        for inst in wo_datasets:
            out_.write(json.dumps(inst))
            out_.write("\n")

if __name__ == "__main__":
    main()