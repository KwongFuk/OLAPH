import json
import argparse

# 读取JSONL文件
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# 计算指定字段的平均值
def calculate_field_averages(data, fields):
    totals = {field: 0 for field in fields}
    count = 0
    for entry in data:
        try:
            # 获取 prediction_scores 中 idx 0 的相关字段
            scores = entry['prediction_scores'][0]
            for field in fields:
                totals[field] += scores.get(field, 0)  # 如果字段不存在则默认为 0
            count += 1
        except (KeyError, IndexError):
            print("缺少 prediction_scores 或指定字段，跳过该条目",entry)
            continue

    if count == 0:
        return {field: 0 for field in fields}
    else:
        return {field: totals[field] / count for field in fields}

def calculate_metrics(averages):
    words_composition = (
        averages['rouge1_f1'] + 
        averages['rouge2_f1'] + 
        averages['rougel_f1']
    )
    semantic_similarity = (
        averages['bleurt'] + 
        averages['bert_score_f1']
    )
    factuality = (
        averages['comprehensive'] - 
        averages['hallucination']
    )
    return words_composition, semantic_similarity, factuality

# 主程序
if __name__ == '__main__':
    # 设置参数解析器
    parser = argparse.ArgumentParser(description="读取 JSONL 文件并计算指定字段的平均值")
    parser.add_argument('input_file', type=str, help="输入的 JSONL 文件路径")
    args = parser.parse_args()

    fields_to_average = [
        "rouge1_p", "rouge1_r", "rouge1_f1", "rouge2_p", "rouge2_r", "rouge2_f1", 
        "rougel_p", "rougel_r", "rougel_f1", "bleurt", "bert_score_p", 
        "bert_score_r", "bert_score_f1", "hallucination", "comprehensive", "fluency"
    ]

    # 读取数据
    data = read_jsonl(args.input_file)

    # 计算各个字段的平均值
    averages = calculate_field_averages(data, fields_to_average)

    print(f"{args.input_file} 中各字段的平均值为:")
    for field, avg in averages.items():
        print(f"{field}: {avg}")
    
    for field, avg in averages.items():    
        print(f"{avg}")

    # 计算并输出三个指标
    words_composition, semantic_similarity, factuality = calculate_metrics(averages)
    print(f"Words Composition: {words_composition/3*100:.1f}")
    print(f"Semantic Similarity: {semantic_similarity/2*100:.1f}")
    print(f"Factuality: {factuality:.1f}")

    print(f"{words_composition/3*100:.1f}")
    print(f"{semantic_similarity/2*100:.1f}")
    print(f"{factuality:.1f}")
