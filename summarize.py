import json
import models
import utils
import os
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from time import time
from tqdm import tqdm


def summarize(examples):
    use_gpu = False
    embed = torch.Tensor(np.load('data/embedding.npz')['embedding'])
    with open("data/word2id.json") as f:
        word2id = json.load(f)
    vocab = utils.Vocab(embed, word2id)
    pred_dataset = utils.Dataset(examples)

    pred_iter = DataLoader(dataset=pred_dataset,
                           batch_size=1,
                           shuffle=False)
    if use_gpu:
        checkpoint = torch.load('checkpoints/CNN_RNN_seed_1.pt')
    else:
        checkpoint = torch.load('checkpoints/CNN_RNN_seed_1.pt', map_location=lambda storage, loc: storage)

    # checkpoint['args']['device'] saves the device used as train time
    # if at test time, we are using a CPU, we must override device to None
    if not use_gpu:
        checkpoint['args'].device = None
    net = getattr(models, checkpoint['args'].model)(checkpoint['args'])
    net.load_state_dict(checkpoint['model'])
    if use_gpu:
        net.cuda()
    net.eval()

    doc_num = len(pred_dataset)
    time_cost = 0
    file_id = 1
    out_str = ""
    for batch in tqdm(pred_iter):
        features, doc_lens = vocab.make_predict_features(batch)
        t1 = time()
        if use_gpu:
            probs = net(Variable(features).cuda(), doc_lens)
        else:
            probs = net(Variable(features), doc_lens)
        t2 = time()
        time_cost += t2 - t1
        start = 0
        for doc_id, doc_len in enumerate(doc_lens):
            stop = start + doc_len
            prob = probs[start:stop]
            topk = min(15, doc_len)
            topk_indices = prob.topk(topk)[1].cpu().data.numpy()
            topk_indices.sort()
            doc = batch[doc_id].split('. ')[:doc_len]
            hyp = [doc[index] for index in topk_indices]
            # with open(os.path.join('outputs/hyp', str(file_id) + '.txt'), 'w') as f:
            # f.write('. '.join(hyp))
            out_str += '. '.join(hyp)
            start = stop
            file_id = file_id + 1
    print('Speed: %.2f docs / s' % (doc_num / time_cost))
    return out_str

# def summarize(input_text):
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     os.chdir(current_dir + '/SummaRuNNer')
#
#     with open('x.txt', 'w') as f:
#         f.write(input_text.lstrip())
#     cmd = ["python", "main.py", "-batch_size", "1", "-predict", "-filename",
#            "x.txt", "-load_dir",
#            "checkpoints/CNN_RNN_seed_1.pt"]
#     subprocess.run(cmd)
#     current_path = os.getcwd()
#     if current_path[-1:-12:-1][::-1] != 'SummaRuNNer':
#         summary = open(f'{current_path}/SummaRuNNer/outputs/hyp/1.txt', 'r').read()
#     else:
#         summary = open(f'{current_path}/outputs/hyp/1.txt', 'r').read()
#
#     # Ideally std::mutex or sys.wait here
#     # Because data race and UB might happen
#     os.chdir(current_dir)
#
#     return str(summary)
