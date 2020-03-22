import os
import time
import random

import yaml

from eval import *
from flask import Flask, jsonify, request, abort
from miscc.config import cfg
from flask_cors import CORS

# from werkzeug.contrib.profiler import ProfilerMiddleware

app = Flask(__name__)
CORS(app)


def create_coco():
    if not request.json or not 'caption' in request.json:
        abort(400)

    caption = request.json['caption']

    t0 = time.time()
    paths = generate_and_save(caption, wordtoix, ixtoword, text_encoder, netG, copies=1)
    base_path = osp.dirname(__file__)
    rel_paths = [osp.relpath(f, base_path) for f in paths]
    t1 = time.time()

    response = {
        'small': rel_paths[0],
        'medium': rel_paths[1],
        'large': rel_paths[2],
        'map1': rel_paths[3],
        'map2': rel_paths[4],
        'caption': caption,
        'elapsed': t1 - t0
    }
    return jsonify(response), 201


if __name__ == '__main__':
    with open('../data/eng_texts.yml', 'r') as f:
        texts = yaml.safe_load(f)
    print(texts)
    sentences_dict = {k: [x + '.' for x in v.split('.')] for k, v in texts.items()}

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

    for name, sentences in sentences_dict:
        for i, sentence in enumerate(sentences):
            paths = generate_and_save(caption, wordtoix, ixtoword, text_encoder, netG, copies=1)
            base_path = osp.dirname(__file__)
            rel_paths = [osp.relpath(f, base_path) for f in paths]
            response = {
                'small': rel_paths[0],
                'medium': rel_paths[1],
                'large': rel_paths[2],
                'map1': rel_paths[3],
                'map2': rel_paths[4],
                'caption': caption,
                'elapsed': t1 - t0
            }
            print(f'done {i}-th text in {name}.')