import os
import os.path as osp
import numpy as np
from PIL import Image
from ISR.models import RDN

if __name__ == '__main__':
    images_dir = r'E:\Projects\digital_writer\AttnGAN\code\images\2020\March\06'
    dirs = os.listdir(images_dir)
    images = [osp.join(images_dir, i, 'coco_g2.png') for i in dirs]
    rdn = RDN(weights='psnr-small')
    for img_path in images:
        print(f'processing {img_path}')
        img = Image.open(img_path)
        lr_img = np.array(img)
        sr_img = rdn.predict(lr_img)
        sr2_img = rdn.predict(sr_img)
        im_res = Image.fromarray(sr2_img)
        im_res.save(osp.join(osp.dirname(img_path), 'coco_g2_big_2.png'), format='png')
        print(f'done processing {img_path}')
