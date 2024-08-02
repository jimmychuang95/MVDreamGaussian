import imageio
from os import listdir
from os.path import isfile, join
import re

def extract_number(s):
    match = re.search(r'\d+', s)
    return int(match.group()) if match else float('inf')

mypath = './outputs/original_clip_result/a_red_panda@20240720-015503/save/it1200-test'
# filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
# filenames.sort()
filenames=[]
for i in range(59, -1, -1):
    print(i)
    filenames.append(f'{i}.png')
for i in range(119, 59, -1):
    print(i)
    filenames.append(f'{i}.png')
print(filenames)
images = []
for filename in filenames:
    images.append(imageio.imread(mypath + '/' + filename))

# 保存为 .mov 文件
imageio.mimwrite(mypath + '/movie.mov', images, fps=30, codec='libx264')
