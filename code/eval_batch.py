import os
import time
import random

import yaml

from eval import *
from miscc.config import cfg
from ISR.models import RDN


if __name__ == '__main__':
    with open('../data/eng_texts.yml', 'r') as f:
        texts = yaml.safe_load(f)
    base_path = osp.dirname(__file__)
    sentences_dict = {k: [x + '.' for x in v.split('.')] for k, v in texts.items()}
    rdn = RDN(weights='psnr-small')

    cfg.CUDA = True

    # load word dictionaries
    wordtoix, ixtoword = word_index()
    # lead models
    text_encoder, netG = models(len(wordtoix))
    # load blob service

    seed = 100
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(seed)

    for name, sentences in sentences_dict.items():
        for i, sentence in enumerate(sentences):
            names2images = generate(sentence, wordtoix, ixtoword, text_encoder, netG, copies=1)
            small_img = names2images['coco_g2']
            sr_img = rdn.predict(small_img)
            sr2_img = rdn.predict(sr_img)
            im_res = Image.fromarray(sr2_img)
            image_path = osp.join(base_path, name, f'{sentence}.png')
            os.makedirs(osp.dirname(image_path), exist_ok=True)
            im_res.save(image_path, format='png')

            print(f'done {i}-th text in {name}.')