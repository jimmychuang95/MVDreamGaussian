import json
import re

json_file = './author_clip/ViT-bigG-14_415_10_random_0.json'

# 讀取 JSON 檔案
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 創建一個 txt 檔案並寫入處理過的 prompt
with open('ViT-bigG-prompts.txt', 'w', encoding='utf-8') as f:
    for key in data.keys():
        if key != "all_mean":  # 忽略 "all_mean"
            prompt = re.sub(r'@\d{8}-\d{6}$', '', key)  # 移除 @後面的數字
            prompt = prompt.replace('_', ' ')  # 將底線替換為空格
            f.write(prompt + '\n')