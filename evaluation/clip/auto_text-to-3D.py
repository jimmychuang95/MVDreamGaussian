import subprocess

prompt_txt = './evaluation/clip/ViT-bigG-prompts.txt'

# 讀取 prompts.txt 檔案
with open(prompt_txt, 'r', encoding='utf-8') as f:
    prompts = f.readlines()

# 移除每行結尾的換行符
prompts = [prompt.strip() for prompt in prompts]

# 遍歷每個 prompt 並運行指令
for prompt in prompts:
    command = [
        'python', 'launch.py',
        '--config', 'configs/gaussiandreamer-sd.yaml',
        '--train',
        '--gpu', '0',
        f'system.prompt_processor.prompt="{prompt}"'
    ]
    print(f'Running command: {command}')
    subprocess.run(command)
