import os
output_files = [
    "/home/nfs02/xingsy/code/output/output_0.txt",
    "/home/nfs02/xingsy/code/output/output_1.txt",
    "/home/nfs02/xingsy/code/output/output_2.txt",
    "/home/nfs02/xingsy/code/output/output_3.txt"
]

output_merged_file = "/home/nfs02/xingsy/code/output/delete_sentence1.txt"

with open(output_merged_file, 'w') as outfile:
    for file_path in output_files:
        with open(file_path, 'r') as infile:
            outfile.write(infile.read())
            
for file_path in output_files:
    os.remove(file_path)

# output_files = [
#     "/home/nfs02/xingsy/code/output/mask/output_0.txt",
#     "/home/nfs02/xingsy/code/output/mask/output_1.txt",
#     "/home/nfs02/xingsy/code/output/mask/output_2.txt",
#     "/home/nfs02/xingsy/code/output/mask/output_3.txt"
# ]

# output_merged_file = "/home/nfs02/xingsy/code/output/mask/merged_output.txt"

# with open(output_merged_file, 'w') as outfile:
#     for file_path in output_files:
#         with open(file_path, 'r') as infile:
#             outfile.write(infile.read())

# for file_path in output_files:
#     os.remove(file_path)

# output_files = [
#     "/home/nfs02/xingsy/code/output/concentrate/output_0.txt",
#     "/home/nfs02/xingsy/code/output/concentrate/output_1.txt",
#     "/home/nfs02/xingsy/code/output/concentrate/output_2.txt",
#     "/home/nfs02/xingsy/code/output/concentrate/output_3.txt"
# ]

# output_merged_file = "/home/nfs02/xingsy/code/output/concentrate/merged_output.txt"

# with open(output_merged_file, 'w') as outfile:
#     for file_path in output_files:
#         with open(file_path, 'r') as infile:
#             outfile.write(infile.read())

# for file_path in output_files:
#     os.remove(file_path)