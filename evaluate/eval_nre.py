from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from collections import defaultdict
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
    return relation_map.get(relation_str, 22)  # Default to 22 if relation_str not found

def count_categories(file_path):
    category_count = {i: 0 for i in range(1, 23)}  # Initialize counts for all categories
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = eval(line.strip())
            relation = ''.join(data['relation'])
            last_slash_index = relation.rfind('/')
            last_part = relation[last_slash_index + 1:] if last_slash_index != -1 else 'None'
            category_number = relation_to_number(last_part)
            category_count[category_number] += 1
    return category_count


def evaluate_metrics(true_file_path, pred_file_path):
    true_labels = []
    predicted_labels = []

    # 读取真实标签文件
    with open(true_file_path, 'r', encoding='utf-8') as true_file:
        for line in true_file:
            data = eval(line.strip())
            relation = ''.join(data['relation'])
            last_slash_index = relation.rfind('/')
            last_part = relation[last_slash_index + 1:] if last_slash_index != -1 else 'None'
            true_labels.append(relation_to_number(last_part))

    # 读取预测结果文件
    with open(pred_file_path, 'r', encoding='utf-8') as pred_file:
        for line in pred_file:
            replace_image_filename, predicted_relation = line.strip().split(' ### ')
            predicted_labels.append(int(predicted_relation))

    # 计算 Precision、Recall、F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
    # 计算准确率
    accuracy = accuracy_score(true_labels, predicted_labels)

    return precision, recall, f1, accuracy

# 示例文件路径
true_file_path = '/home/nfs02/xingsy/code/data/MNRE/mnre_txt/mnre_test.txt'
pred_file_path = '/home/nfs02/xingsy/code/output/merged_output.txt'

# 统计类别数量
category_count = count_categories(true_file_path)
# category_c = count_categories_from_pred_file(pred_file_path)
print("Category counts in true file:")
c = 0
for category, count in category_count.items():
    print(f"Category {category}: {count}")
    c += count

#计算 Precision、Recall、F1-score 和 准确率
precision, recall, f1, accuracy = evaluate_metrics(true_file_path, pred_file_path)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
print(f"Accuracy: {accuracy}")
