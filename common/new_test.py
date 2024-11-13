
from collections import defaultdict

import matplotlib.pyplot as plt
import tqdm
    # 提取数据
def plot_category_counts(base_counts, wrong_in_pred_counts):
    # 提取数据
    categories_base = list(base_counts.keys())
    counts_base = list(base_counts.values())
    
    categories_wrong = list(wrong_in_pred_counts.keys())
    counts_wrong = list(wrong_in_pred_counts.values())
    
    # 合并两个数据集的类别，以便创建全面的 x 轴标签
    all_categories = set(categories_base) | set(categories_wrong)
    all_categories = sorted(all_categories)
    
    base_values = [base_counts.get(cat, 0) for cat in all_categories]
    wrong_values = [wrong_in_pred_counts.get(cat, 0) for cat in all_categories]
    
    # 创建一个新的图形
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 设置柱状图的位置
    x = range(len(all_categories))
    width = 0.35  # 柱状图宽度
    
    # 绘制 base_counts 的柱状图
    ax.bar([p - width/2 for p in x], base_values, width=width, color='blue', alpha=0.6, label='Base File Correct Counts')
    
    # 绘制 wrong_in_pred_counts 的柱状图
    ax.bar([p + width/2 for p in x], wrong_values, width=width, color='red', alpha=0.6, label='Wrong Predictions Counts')
    
    # 设置标签和标题
    ax.set_xlabel('Category')
    ax.set_ylabel('Count')
    ax.set_title('Category Counts in Base File and Wrong Predictions')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{cat}' for cat in all_categories], rotation=90)
    
    # 添加图例
    ax.legend()
    
    # 显示图形
    plt.tight_layout()
    save_path = '/home/nfs02/xingsy/code/playground/temp.png'
    plt.savefig(save_path)
    plt.close()

def relation_to_number(relation_str):
    relation_map = {
        'race': 1,
        'place_of_birth': 2,
        'present_in': 3,
        'alternate_names': 4,
        'parent': 5,
        'locate_at': 6,
        'neighbor': 7,
        'part_of': 8,
        'alumni': 9,
        'siblings': 10,
        'religion': 11,
        'place_of_residence': 12,
        'nationality': 13,
        'contain': 14,
        'couple': 15,
        'awarded': 16,
        'subsidiary': 17,
        'peer': 18,
        'held_on': 19,
        'member_of': 20,
        'charges': 21,
        'None': 22
    }
    return relation_map.get(relation_str, 22)  

def load_labels(file_path, is_predicted=False, mask= False):
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if mask:
                _, predicted_relation = line.strip().split(' ### ')
                labels.append(int(predicted_relation))

            elif is_predicted:
                _, _, predicted_relation = line.strip().split(' ### ')
                labels.append(int(predicted_relation))
            else:
                data = eval(line.strip())
                relation = ''.join(data['relation'])
                last_slash_index = relation.rfind('/')
                last_part = relation[last_slash_index + 1:] if last_slash_index != -1 else 'None'
                labels.append(relation_to_number(last_part))
    return labels

def convert_keys_to_relation(counts_dict):
    relation_map = {
        'race': 1,
        'place_of_birth': 2,
        'present_in': 3,
        'alternate_names': 4,
        'parent': 5,
        'locate_at': 6,
        'neighbor': 7,
        'part_of': 8,
        'alumni': 9,
        'siblings': 10,
        'religion': 11,
        'place_of_residence': 12,
        'nationality': 13,
        'contain': 14,
        'couple': 15,
        'awarded': 16,
        'subsidiary': 17,
        'peer': 18,
        'held_on': 19,
        'member_of': 20,
        'charges': 21,
        'None': 22
    }
    reverse_relation_map = {v: k for k, v in relation_map.items()}
    return {reverse_relation_map.get(key, 'Unknown'): value for key, value in counts_dict.items()}

def convert_key_to_relation_string(relation_id):
    relation_map = {
        1: 'race',
        2: 'place_of_birth',
        3: 'present_in',
        4: 'alternate_names',
        5: 'parent',
        6: 'locate_at',
        7: 'neighbor',
        8: 'part_of',
        9: 'alumni',
        10: 'siblings',
        11: 'religion',
        12: 'place_of_residence',
        13: 'nationality',
        14: 'contain',
        15: 'couple',
        16: 'awarded',
        17: 'subsidiary',
        18: 'peer',
        19: 'held_on',
        20: 'member_of',
        21: 'charges',
        22: 'None'
    }
    return relation_map.get(relation_id, 'Unknown')

def find_correct_in_base_but_wrong_in_pred(true_file_path, base_file_path, pred_file_path):
    true_labels = load_labels(true_file_path)
    base_labels = load_labels(base_file_path, is_predicted=True)
    pred_labels = load_labels(pred_file_path, mask=True)
    
    
    correct_in_base = {i for i, (true, pred) in enumerate(zip(true_labels, base_labels)) if true != pred}
    incorrect_in_pred = {i for i, (true, pred) in enumerate(zip(true_labels, pred_labels)) if true == pred}
    
    # print(incorrect_in_pred)
    # correct_in_base = {i for i in correct_in_base if true_labels[i] != 22}
    # incorrect_in_pred = {i for i in incorrect_in_pre if true_labels[i] != 22}

    # # Recalculate correct_in_base and incorrect_in_pred considering only the first 100 samples
    # correct_in_base = {i for i, (true, pred) in enumerate(zip(true_labels[:limit], base_labels[:limit])) if true == pred}
    # incorrect_in_pred = {i for i, (true, pred) in enumerate(zip(true_labels[:limit], pred_labels[:limit])) if true != pred}
        
    result_counts = defaultdict(int)
    category_samples = defaultdict(list)
    index = -1
    output_file_path = '/home/nfs02/xingsy/code/mask_test.txt'
    file_path = '/home/nfs02/xingsy/code/data/MNRE/mnre_txt/mnre_test.txt'
    with open(output_file_path, 'w', encoding='utf-8') as out_file:
        i = -1
        with open(true_file_path, 'r', encoding='utf-8') as file:
            for line in ((file)):
                
                # 解析每一行的JSON数据
                i += 1
                data = eval(line.strip())

                sentence = ' '.join(data['token'])
                head = data['h']['name']
                tail = data['t']['name']
                image = data['img_id']
                if i in correct_in_base and i in incorrect_in_pred:
                    label = true_labels[i]
                    result_counts[label] += 1

                    # 获取当前的head, tail, relation值
                    original_relation = convert_key_to_relation_string(base_labels[i])
                    new_relation = convert_key_to_relation_string(pred_labels[i])
                    # 将信息写入文件
                    out_file.write(f"Line {i}:, Sentence: {sentence},  Image: {image}, Head: {head}, Tail: {tail}, delete: {original_relation}, base: {new_relation}\n")
    
    return result_counts

true_file_path = '/home/nfs02/xingsy/code/data/MNRE/mnre_txt/mnre_test.txt'
base_file_path = '/home/nfs02/xingsy/code/output/delete_image.txt'
pred_file_path = '/home/nfs02/xingsy/code/output/aligngpt-7b-finetune7epochnew.txt'


wrong_in_pred = find_correct_in_base_but_wrong_in_pred(true_file_path, base_file_path, pred_file_path)

true_labels = load_labels(true_file_path)
base_labels = load_labels(base_file_path, is_predicted=True)
correct_in_base = {i for i, (true, pred) in enumerate(zip(true_labels, base_labels)) if true == pred}

base_counts = defaultdict(int)
for i in (correct_in_base):
    base_counts[true_labels[i]] += 1
    
base_counts = convert_keys_to_relation(base_counts)
wrong_in_pred = convert_keys_to_relation(wrong_in_pred)
print("correct in base", base_counts)
print("wrong in pred", wrong_in_pred)
sum = 0
for i in wrong_in_pred.values():
    sum += i
print("all wrong is",sum)
plot_category_counts(base_counts, wrong_in_pred)