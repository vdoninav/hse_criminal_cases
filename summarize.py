import os
import subprocess


def summarize(input_text):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir + '/SummaRuNNer')

    with open('x.txt', 'w') as f:
        f.write(input_text.lstrip())
    cmd = ["python", "main.py", "-batch_size", "1", "-predict", "-filename",
           "x.txt", "-load_dir",
           "checkpoints/CNN_RNN_seed_1.pt"]
    subprocess.run(cmd)
    current_path = os.getcwd()
    if current_path[-1:-12:-1][::-1] != 'SummaRuNNer':
        summary = open(f'{current_path}/SummaRuNNer/outputs/hyp/1.txt').read()
    else:
        summary = open(f'{current_path}/outputs/hyp/1.txt').read()
    os.chdir(current_dir)

    return str(summary)
