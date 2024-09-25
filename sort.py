import json

# 读取JSONL文件
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# 按照 prediction_scores 中 idx 0 的 comprehensive 字段排序
def sort_by_comprehensive(data):
    return sorted(data, key=lambda x: x['prediction_scores'][0]['comprehensive'], reverse=False)

# 保存排序后的结果为新的JSONL文件
def save_jsonl(data, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# 主程序
if __name__ == '__main__':
    input_file = 'pdata_llama3-8b-instruct_live_qa_sampling.jsonl'  # 输入文件名
    output_file = 'sorted_pdata_llama3-8b-instruct_live_qa_sampling.jsonl'  # 输出文件名

    # 读取数据
    data = read_jsonl(input_file)

    # 根据 comprehensive 字段排序
    sorted_data = sort_by_comprehensive(data)

    # 保存排序后的数据
    save_jsonl(sorted_data, output_file)

    print(f"排序后的数据已保存至 {output_file}")
