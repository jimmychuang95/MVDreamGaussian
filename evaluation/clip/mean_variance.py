import json
import numpy as np

# 讀取所有 JSON 文件並提取 all_mean
all_means = []
for i in range(10):
    filename = f'./evaluation/clip_result/GaussianDreamer-{i}.json'
    with open(filename, 'r') as file:
        data = json.load(file)
        all_means.append(data['all_mean'])

# 計算平均值和方差
mean_all_means = np.mean(all_means)
variance_all_means = np.var(all_means)
std_all_means = np.std(all_means)

print(f'平均值: {mean_all_means}')
print(f'方差: {variance_all_means}')
print(f'標準差: {std_all_means}')
