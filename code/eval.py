from __future__ import print_function

import os
import sys
import os.path as osp
import torch
import io
import time
import numpy as np
from PIL import Image
import torch.onnx
from datetime import datetime
from torch.autograd import Variable
from miscc.config import cfg, cfg_from_file
from miscc.utils import build_super_images2
from model import RNN_ENCODER, G_NET
import tensorflow as tf
from ISR.models import RDN

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from cachelib import SimpleCache

cache = SimpleCache()

cfg_from_file(r'E:\Projects\digital_writer\AttnGAN\code\cfg\eval_coco.yml')

# otherwise it allocates all memory and no memory is left for pytorch
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
rdn = RDN(weights='psnr-large')
# rdn = RDN(weights='psnr-small')


def vectorize_caption(wordtoix, caption, copies=2):
    # create caption vector
    tokens = caption.split(' ')
    cap_v = []
    for t in tokens:
        t = t.strip().encode('ascii', 'ignore').decode('ascii')
        if len(t) > 0 and t in wordtoix:
            cap_v.append(wordtoix[t])

    # expected state for single generation
    captions = np.zeros((copies, len(cap_v)))
    for i in range(copies):
        captions[i, :] = np.array(cap_v)
    cap_lens = np.zeros(copies) + len(cap_v)

    # print(captions.astype(int), cap_lens.astype(int))
    # captions, cap_lens = np.array([cap_v, cap_v]), np.array([len(cap_v), len(cap_v)])
    # print(captions, cap_lens)
    # return captions, cap_lens

    return captions.astype(int), cap_lens.astype(int)


def generate(caption, wordtoix, ixtoword, text_encoder, netG, copies=2):
    # load word vector
    captions, cap_lens = vectorize_caption(wordtoix, caption, copies)

    # only one to generate
    batch_size = captions.shape[0]

    nz = cfg.GAN.Z_DIM
    with torch.no_grad():
        captions = Variable(torch.from_numpy(captions))
        cap_lens = Variable(torch.from_numpy(cap_lens))
        noise = Variable(torch.FloatTensor(batch_size, nz))

    if cfg.CUDA:
        captions = captions.cuda()
        cap_lens = cap_lens.cuda()
        noise = noise.cuda()

    #######################################################
    # (1) Extract text embeddings
    #######################################################
    hidden = text_encoder.init_hidden(batch_size)
    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
    mask = (captions == 0)

    #######################################################
    # (2) Generate fake images
    #######################################################
    noise.data.normal_(0, 1)
    fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)

    # G attention
    cap_lens_np = cap_lens.cpu().data.numpy()

    names2images = {}

    for j in range(batch_size):
        for k in range(len(fake_imgs)):
            im = fake_imgs[k][j].data.cpu().numpy()
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            im = np.transpose(im, (1, 2, 0))
            im = rdn.predict(im)
            im = rdn.predict(im)

            # save image to stream
            if copies > 2:
                blob_name = osp.join(str(j), f'coco_g{k}.png')
            else:
                blob_name = f'coco_g{k}.png'
            names2images[blob_name] = im

        for k in range(len(attention_maps)):
            if len(fake_imgs) > 1:
                im = fake_imgs[k + 1].detach().cpu()
            else:
                im = fake_imgs[0].detach().cpu()

            attn_maps = attention_maps[k]
            att_sze = attn_maps.size(2)

            img_set, sentences = \
                build_super_images2(im[j].unsqueeze(0),
                                    captions[j].unsqueeze(0),
                                    [cap_lens_np[j]], ixtoword,
                                    [attn_maps[j]], att_sze)

            if img_set is not None:
                blob_name = f'attmaps_a{k}.png'
                names2images[blob_name] = img_set

    return names2images


def generate_and_save(caption, wordtoix, ixtoword, text_encoder, netG, copies=2):
    prefix = osp.join(osp.dirname(__file__), 'images', datetime.now().strftime('%Y/%B/%d/%H_%M_%S_%f'))
    names2images = generate(caption, wordtoix, ixtoword, text_encoder, netG, copies)

    # load word vector
    paths = []
    for name, image in names2images.items():
        im = Image.fromarray(image)
        blob_name = osp.join(prefix, name)
        os.makedirs(osp.dirname(blob_name), exist_ok=True)
        im.save(blob_name, format="png")
        paths.append(blob_name)

    return paths


def word_index():
    ixtoword = cache.get('ixtoword')
    wordtoix = cache.get('wordtoix')
    if ixtoword is None or wordtoix is None:
        # print("ix and word not cached")
        # load word to index dictionary
        x = pickle.load(open(osp.join(cfg.DATA_DIR, 'captions.pickle'), 'rb'))
        ixtoword = x[2]
        wordtoix = x[3]
        del x
        cache.set('ixtoword', ixtoword, timeout=60 * 60 * 24)
        cache.set('wordtoix', wordtoix, timeout=60 * 60 * 24)

    return wordtoix, ixtoword


def models(word_len):
    # print(word_len)
    text_encoder = cache.get('text_encoder')
    if text_encoder is None:
        # print("text_encoder not cached")
        text_encoder = RNN_ENCODER(word_len, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        if cfg.CUDA:
            text_encoder.cuda()
        text_encoder.eval()
        cache.set('text_encoder', text_encoder, timeout=60 * 60 * 24)

    netG = cache.get('netG')
    if netG is None:
        # print("netG not cached")
        netG = G_NET()
        state_dict = torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
        netG.load_state_dict(state_dict)
        if cfg.CUDA:
            netG.cuda()
        netG.eval()
        cache.set('netG', netG, timeout=60 * 60 * 24)

    return text_encoder, netG


def eval(caption):
    # load word dictionaries
    wordtoix, ixtoword = word_index()
    # lead models
    text_encoder, netG = models(len(wordtoix))
    # load blob service
    t0 = time.time()
    generate_and_save(caption, wordtoix, ixtoword, text_encoder, netG)
    t1 = time.time()


if __name__ == "__main__":
    caption = "Light intensifies, blinds, contours dissolve, fades like faces in an old photograph"

    # load configuration
    # cfg_from_file('eval_bird.yml')
    # load word dictionaries
    wordtoix, ixtoword = word_index()
    # lead models
    text_encoder, netG = models(len(wordtoix))
    # load blob service

    t0 = time.time()
    urls = generate_and_save(caption, wordtoix, ixtoword, text_encoder, netG)
    t1 = time.time()
    print(t1 - t0)
    print(urls)
