import os
import numpy as np
import skimage.draw
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import datetime
import mmcv

def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

def main():
    image_path = 'E:/2023/dataset/balloon/val/3825919971_93fb1ec581_b.jpg'
    print("Running on {}".format(image_path))
    image = skimage.io.imread(image_path)
    gray = skimage.color.rgb2gray(image)   #debug
    gray = skimage.color.gray2rgb(gray) * 255  #debug
    splash = gray.astype(np.uint8)   #debug
    skimage.io.imsave('splash_test.png', splash)
    config_file = 'E:/2023/mmlab/mmdetection-master/tools/work_dirs/mask_rcnn_r50_fpn_1x_wandb_balloon_coco_pretrain/mask_rcnn_r50_fpn_1x_wandb_coco.py'
    checkpoint_file = 'E:/2023/mmlab/mmdetection-master/tools/work_dirs/mask_rcnn_r50_fpn_1x_wandb_balloon_coco_pretrain/epoch_24.pth'
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    # test a single image
    result = inference_detector(model, image)
    bbox_result, segm_result = result
    segms=segm_result[0]
    segms = np.stack(segms, axis=2)
    splash = color_splash(image, segms)
    # Save output
    file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    skimage.io.imsave(file_name, splash)

    show_result_pyplot(model, image, result)

if __name__ == '__main__':
    main()
