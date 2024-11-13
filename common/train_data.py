import json

# 文件路径
file_path = '/home/nfs02/xingsy/code/data/MNRE/mnre_txt/mnre_train.txt'
output_file = '/home/nfs02/xingsy/code/data/MNRE/train.json'

prompt = '''Input sentence: {}
Head: {}
Tail: {}

The known relationships are [race, place_of_birth, present_in, alternate_names, parent, locate_at, neighbor, part_of, alumni, siblings, religion, place_of_residence, nationality, contain, couple, awarded, subsidiary, peer, held_on, member_of, charges, None]. 

Given the input sentence, 'head', and 'tail', identify the appropriate relationship between 'head' and 'tail' from the above list. The 'head' and 'tail' are provided as input entities. 

Select the relationship that best describes the connection between 'head' and 'tail' based on the provided sentence and output the result in JSON format with 'relation', 'head', and 'tail'. 

For example:
Output: ['relation': '', 'head': '', 'tail': '']
 '''
# 读取文件内容
with open(file_path, 'r') as file:
    lines = file.readlines()

# 处理每一行并构造数据集
datasets = []
for idx, line in enumerate(lines):
    # 这里假设每一行的数据格式为 JSON 字符串
    line_data = eval(line.strip())
    
    # 提取数据
    extracted_data = {
        'token': line_data.get('token', []),
        'h': line_data.get('h', {'name': '', 'pos': []}),
        't': line_data.get('t', {'name': '', 'pos': []}),
        'img_id': line_data.get('img_id', ''),
        'relation': line_data.get('relation', '')
    }
    prompt = (
        f"Input sentence: {' '.join(extracted_data['token'])}\n"
        f"Head: {extracted_data['h']['name']}\n"
        f"Tail: {extracted_data['t']['name']}\n\n"
        f"The known relationships are [race, place_of_birth, present_in, alternate_names, parent, locate_at, neighbor, part_of, alumni, siblings, religion, place_of_residence, nationality, contain, couple, awarded, subsidiary, peer, held_on, member_of, charges, None]. \n\n"
        f"Given the input sentence, 'head', and 'tail', identify the appropriate relationship between 'head' and 'tail' from the above list. The 'head' and 'tail' are provided as input entities. \n\n"
        f"Select the relationship that best describes the connection between 'head' and 'tail' based on the provided sentence and output the result in JSON format with 'relation', 'head', and 'tail'. \n\n"

    )
    
    # 构造对话格式数据集
    data = {
        "id": f"example_id_{idx+1}",  # 生成唯一的 ID
        "image": extracted_data['img_id'],
        "conversations": [
            {
                "from": "human",
                "value": prompt
            },
            {
                "from": "gpt",
                "value": f"Output: ['relation': '{extracted_data['relation'].split('/')[-1]}', 'head': '{extracted_data['h']['name']}', 'tail': '{extracted_data['t']['name']}']"
            }
        ]
    }
    
    datasets.append(data)

# 输出 JSON 数据到文件
with open(output_file, 'w') as file:
    json.dump(datasets, file, indent=4)

print(f"JSON 数据已成功生成并保存到 '{output_file}'")
