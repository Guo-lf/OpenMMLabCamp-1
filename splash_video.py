import cv2
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
    video_path='E:/2023/mmlab_cmp/test_video.mp4'
    config_file = 'E:/2023/mmlab/mmdetection-master/tools/work_dirs/mask_rcnn_r50_fpn_1x_wandb_balloon_coco_pretrain/mask_rcnn_r50_fpn_1x_wandb_coco.py'
    checkpoint_file = 'E:/2023/mmlab/mmdetection-master/tools/work_dirs/mask_rcnn_r50_fpn_1x_wandb_balloon_coco_pretrain/epoch_24.pth'
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    # Video capture
    vcapture = cv2.VideoCapture(video_path)
    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vcapture.get(cv2.CAP_PROP_FPS)

    # Define codec and create video writer
    file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
    vwriter = cv2.VideoWriter(file_name,
                              cv2.VideoWriter_fourcc(*'MJPG'),
                              fps, (width, height))
    count = 0
    success = True
    while success:
        print("frame: ", count)
        # Read next image
        success, image = vcapture.read()
        if success:
            # OpenCV returns images as BGR, convert to RGB
            image = image[..., ::-1]
            # Detect objects
            result = inference_detector(model, image)
            bbox_result, segm_result = result
            segms = segm_result[0]
            segms = np.stack(segms, axis=2)
            # Color splash
            splash = color_splash(image, segms)
            # RGB -> BGR to save image to video
            splash = splash[..., ::-1]
            # Add image to video writer
            vwriter.write(splash)
            count += 1
    vwriter.release()
    print("Saved to ", file_name)


if __name__ == '__main__':
    main()