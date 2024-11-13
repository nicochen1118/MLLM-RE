
# find the number of relation
ans = set()
with open('/home/nfs02/xingsy/code/data/MNRE/mnre_txt/mnre_train.txt', 'r', encoding='utf-8') as file:
    for line in file:
        # 解析每一行的JSON数据
        data = eval(line.strip())
        # 处理文本数据
        relation = ''.join(data['relation'])
        # 找到最后一个斜杠的位置
        last_slash_index = relation.rfind('/')
        last_part = ''
        # 取斜杠后的字符
        if last_slash_index != -1:
            last_part = relation[last_slash_index + 1:]
            ans.add(last_part) 
        else:
            print("No slash found in the string.", relation)
print(len(ans))
for part in ans:
    print(part)