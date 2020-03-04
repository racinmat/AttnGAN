import os
import time
import random
from eval import *
from flask import Flask, jsonify, request, abort
from miscc.config import cfg

# from werkzeug.contrib.profiler import ProfilerMiddleware

app = Flask(__name__)


@app.route('/api/v1.0/coco', methods=['POST'])
def create_coco():
    if not request.json or not 'caption' in request.json:
        abort(400)

    caption = request.json['caption']

    t0 = time.time()
    urls = generate(caption, wordtoix, ixtoword, text_encoder, netG)
    t1 = time.time()

    response = {
        'small': urls[0],
        'medium': urls[1],
        'large': urls[2],
        'map1': urls[3],
        'map2': urls[4],
        'caption': caption,
        'elapsed': t1 - t0
    }
    return jsonify({'bird': response}), 201


@app.route('/', methods=['GET'])
def get_coco():
    return 'Version 1'


if __name__ == '__main__':
    t0 = time.time()

    # gpu based
    cfg.CUDA = os.environ["GPU"].lower() == 'true'

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

    app.run(host='0.0.0.0', port=8080)
