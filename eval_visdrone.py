
from models.meta_architectures import BuildFrameworks
import tensorflow as tf
import cv2
import os
import time
import numpy as np
from core.tools import tools
from core.config import cfgs


def txt_visdrone(img_dir, save_path, ensemble_path,normalized_result=False):
    # initialize
    net = BuildFrameworks(False)
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])

    if cfgs.PRETRAIN_BACKBONE_NAME in ['resnet101_v1d', 'resnet50_v1d']:
        img_tensor = img_plac / 255 - tf.constant([[cfgs.PIXEL_MEAN_]])
        img_tensor = img_tensor / tf.constant([cfgs.PIXEL_STD])
    else:
        img_tensor = img_plac - tf.constant([[cfgs.PIXEL_MEAN]])
    img_tensor = tf.expand_dims(img_tensor, axis=0)

    # detection result dict
    det_dict = net.build_framworks(input_img_batch=img_tensor)
    final_boxes, final_scores, final_category = det_dict[
                                                    "final_boxes"], det_dict["final_scores"], det_dict["final_category"]

    restorer, restore_ckpt = net.get_restorer()
    if restore_ckpt == cfgs.PRETRAIN_BACKBONE_WEIGHTS:
        raise IndexError("----->Please select the trained weights rather than pretrained weights.")
    img_name = os.listdir(path=img_dir)
    with tf.Session() as sess:
        restorer.restore(sess, restore_ckpt)
        for i in range(len(img_name)):
            # read images
            img_array = cv2.imread(os.path.join(img_dir, img_name[i]))
            raw_h = img_array.shape[0]
            raw_w = img_array.shape[1]
            detected_scores_, detected_boxes_, detected_categories_ = [], [], []
            start=time.time()
            if cfgs.MULTI_SCALSE_TESTING:
                short_size_len=cfgs.SHORT_SIZE_LIMITATION_LIST
            else:
                short_size_len=[cfgs.SHORT_SIZE_LIMITATION]
            for short_size in short_size_len:
                if raw_h < raw_w:
                    new_h, new_w = short_size, min(int(short_size * float(raw_w) / raw_h), cfgs.MAX_LENGTH)
                else:
                    new_h, new_w = min(int(short_size * float(raw_h) / raw_w), cfgs.MAX_LENGTH), short_size
                img_resize = cv2.resize(img_array, (new_w, new_h))
                resized_img_, final_boxes_, final_scores_, final_category_ = sess.run(
                    [img_plac, final_boxes, final_scores, final_category],
                    feed_dict={img_plac: img_resize[:, :, ::-1]})

                #剔除变成一条线的框
                inds_inside = np.where(
                    (final_boxes_[:, 2] > final_boxes_[:, 0]) &
                    (final_boxes_[:, 3] > final_boxes_[:, 1])
                )[0]

                final_boxes_ = final_boxes_[inds_inside, :]
                final_scores_ = final_scores_[inds_inside]
                final_category_ = final_category_[inds_inside]

                xmin, ymin, xmax, ymax = final_boxes_[:, 0], final_boxes_[:, 1], \
                                         final_boxes_[:, 2], final_boxes_[:, 3]

                resized_h, resized_w = resized_img_.shape[0], resized_img_.shape[1]

                # normalized boxes
                xmin = xmin / resized_w
                xmax = xmax / resized_w
                ymin = ymin / resized_h
                ymax = ymax / resized_h

                resized_boxes = np.transpose(np.stack([xmin, ymin, xmax, ymax]))

                detected_scores_.append(final_scores_)
                detected_boxes_.append(resized_boxes)
                detected_categories_.append(final_category_)
            end=time.time()

            from ensemble_boxes import weighted_boxes_fusion

            boxes, scores, labels = weighted_boxes_fusion(detected_boxes_, detected_scores_, detected_categories_,
                                                          weights=None,
                                                          iou_thr=0.6, skip_box_thr=0.001, conf_type='avg')

            detected_scores = np.array(scores)
            detected_boxes = np.array(boxes)
            detected_categories = np.array(labels)
            if not normalized_result:
                # map coordinate to raw picture
                xmin, ymin, xmax, ymax = detected_boxes[:, 0] * raw_w, detected_boxes[:, 1] * raw_h, \
                                         detected_boxes[:, 2] * raw_w, detected_boxes[:, 3] * raw_h
                boxes = np.transpose(np.stack([xmin, ymin, xmax - xmin, ymax - ymin]))
                result_full_path = save_path + \
                                   str('/' + img_name[i].split('.')[0]) + '.txt'
            else:
                result_full_path = ensemble_path +\
                                   str('/' + img_name[i].split('.')[0]) + '.txt'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            if not os.path.exists(ensemble_path):
                os.makedirs(ensemble_path)
            with open(result_full_path, 'w') as fout:
                num = 0
                if normalized_result:
                    fout.write(str(raw_w))
                    fout.write(',')
                    fout.write(str(raw_h))
                    fout.write('\n')
                for j, box in enumerate(boxes):
                    if num < 500:
                        if detected_categories[j] != 1 and detected_categories[j] != 12:
                            fout.write(str(box[0]))
                            fout.write(',')
                            fout.write(str(box[1]))
                            fout.write(',')
                            fout.write(str(box[2]))
                            fout.write(',')
                            fout.write(str(box[3]))
                            fout.write(',')
                            fout.write(str(detected_scores[j]))
                            fout.write(',')
                            fout.write(str(detected_categories[j] - 1))
                            fout.write(',-1,')
                            fout.write('-1' + '\n')
                            num += 1
                    else:
                        break
                fout.close()
            tools.view_bar(
                '{}, time cost: {}s'.format(
                    img_name[i],
                    (end - start)),
                i + 1,
                len(img_name))


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    img_dir = cfgs.IMG_INFERENCE_DIR
    save_path = r'./test_result_txt'
    ensemble_path = './test_model_ensemble' + '/' + cfgs.PRETRAIN_BACKBONE_NAME +'/'
    #if you want to get nomalized results for  model ensemble, set normalized_result = True
    txt_visdrone(img_dir, save_path,ensemble_path,normalized_result=cfgs.NORMALIZED_RESULTS_FOR_MODEL_ENSEMBLE)
