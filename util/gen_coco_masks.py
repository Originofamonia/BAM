import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from PIL import Image
from tqdm import tqdm


def generate_masks(ann_file, save_dir, prefix):
    os.makedirs(save_dir, exist_ok=True)
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()

    for img_id in tqdm(img_ids, desc=f"Processing {prefix}"):
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(img_id)[0]
        height, width = img_info['height'], img_info['width']
        mask = np.zeros((height, width), dtype=np.uint8)

        for ann in anns:
            cat_id = ann['category_id']
            rle = coco.annToRLE(ann)
            m = maskUtils.decode(rle)
            mask[m == 1] = cat_id  # or ann['id'] for instance mask

        file_stem = img_info['file_name'].split('_')[-1].split('.')[0]
        out_path = os.path.join(save_dir, f"{prefix}_{file_stem}.png")
        Image.fromarray(mask).save(out_path)


def main():
    # === Paths ===
    coco_root = '/home/edward/data/MSCOCO2014/annotations'

    # Train
    train_ann_file = os.path.join(coco_root, 'instances_train2014.json')
    train_save_dir = os.path.join(coco_root, 'train2014')
    generate_masks(train_ann_file, train_save_dir, 'COCO_train2014')

    # Val
    val_ann_file = os.path.join(coco_root, 'instances_val2014.json')
    val_save_dir = os.path.join(coco_root, 'val2014')
    generate_masks(val_ann_file, val_save_dir, 'COCO_val2014')


if __name__ == '__main__':
    main()
