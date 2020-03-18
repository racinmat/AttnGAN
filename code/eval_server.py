import os
import time
import random
from eval import *
from flask import Flask, jsonify, request, abort
from miscc.config import cfg
from flask_cors import CORS

# from werkzeug.contrib.profiler import ProfilerMiddleware

app = Flask(__name__)
CORS(app)


@app.route('/api/v1.0/coco', methods=['POST'])
def create_coco():
    if not request.json or not 'caption' in request.json:
        abort(400)

    caption = request.json['caption']

    t0 = time.time()
    paths = generate(caption, wordtoix, ixtoword, text_encoder, netG, copies=1)
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


@app.route('/', methods=['GET'])
def get_coco():
    return 'Version 1'


if __name__ == '__main__':
    t0 = time.time()

    # gpu based
    # cfg.CUDA = os.environ["GPU"].lower() == 'true'
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

    app.run(host='0.0.0.0', port=8088)
