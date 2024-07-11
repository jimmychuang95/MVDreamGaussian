import imageio
from os import listdir
from os.path import isfile, join
import re

def extract_number(s):
    match = re.search(r'\d+', s)
    return int(match.group()) if match else float('inf')

mypath = './outputs/gaussiandreamer-sd/sniper_rifle@20240702-211435/test'
filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
filenames.sort(key=extract_number)
print(filenames)
images = []
for filename in filenames:
    images.append(imageio.imread(mypath + '/' + filename))

# 保存为 .mov 文件
imageio.mimwrite(mypath + '/movie.mov', images, fps=30, codec='libx264')
