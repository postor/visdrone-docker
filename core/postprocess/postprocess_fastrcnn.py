
import tensorflow as tf
from core.config import cfgs
from core.postprocess.boxes_process import clip_boxes_to_img_boundaries
from core.box_utils import encode_and_decode

def postprocess_fastrcnn(rois, bbox_ppred, scores, img_shape,is_training):
    '''
    :param rois:[-1, 4]
    :param bbox_ppred: [-1, (cfgs.Class_num+1) * 4]
    :param scores: [-1, cfgs.Class_num + 1]
    :return:
    '''
    rois = tf.stop_gradient(rois)
    scores = tf.stop_gradient(scores)
    bbox_ppred = tf.reshape(bbox_ppred, [-1, cfgs.CLASS_NUM + 1, 4])
    bbox_ppred = tf.stop_gradient(bbox_ppred)

    bbox_pred_list = tf.unstack(bbox_ppred, axis=1)
    score_list = tf.unstack(scores, axis=1)

    allclasses_boxes = []
    allclasses_scores = []
    categories = []
    for i in range(1, cfgs.CLASS_NUM + 1):
        # 1. decode boxes in each class
        tmp_encoded_box = bbox_pred_list[i]
        tmp_score = score_list[i]
        tmp_decoded_boxes = encode_and_decode.decode_boxes(encoded_boxes=tmp_encoded_box,
                                                           reference_boxes=rois,
                                                           scale_factors=cfgs.ROI_SCALE_FACTORS[-1])
        # tmp_decoded_boxes = encode_and_decode.decode_boxes(boxes=rois,
        #                                                    deltas=tmp_encoded_box,
        #                                                    scale_factor=cfgs.ROI_SCALE_FACTORS)

        # 2. clip to img boundaries
        tmp_decoded_boxes = clip_boxes_to_img_boundaries(decode_boxes=tmp_decoded_boxes,
                                                                     img_shape=img_shape)

        # 3. NMS
        # import time
        # start_time=time.time()
        keep = tf.image.non_max_suppression(
            boxes=tmp_decoded_boxes,
            scores=tmp_score,
            max_output_size=cfgs.RCNN_NMS_MAX_BOXES_PER_CLASS,
            iou_threshold=cfgs.RCNN_NMS_IOU_THR)
        # end_time = time.time()
        # print(end_time - start_time)
        perclass_boxes = tf.gather(tmp_decoded_boxes, keep)
        perclass_scores = tf.gather(tmp_score, keep)

        allclasses_boxes.append(perclass_boxes)
        allclasses_scores.append(perclass_scores)
        categories.append(tf.ones_like(perclass_scores) * i)

    final_boxes = tf.concat(allclasses_boxes, axis=0)
    final_scores = tf.concat(allclasses_scores, axis=0)
    final_category = tf.concat(categories, axis=0)

    if is_training:
        #summary
        kept_indices = tf.reshape(tf.where(tf.greater_equal(final_scores, cfgs.SUMMARY_SCORE_THR)), [-1])
        final_boxes = tf.gather(final_boxes, kept_indices)
        final_scores = tf.gather(final_scores, kept_indices)
        final_category = tf.gather(final_category, kept_indices)


    det_dict={
        "final_boxes":final_boxes,
        "final_scores":final_scores,
        "final_category":final_category
    }


    return det_dict

def postprocess_cascadercnn(rois, bbox_pred, cls_prob, stage):
    '''
    :param rois:[-1, 4]
    :param bbox_ppred: bbox_ppred: [-1, 4]
    :param scores: [-1, 1]
    :return:
    '''
    # rois = tf.stop_gradient(rois)
    # bbox_pred = tf.stop_gradient(bbox_pred)
    bbox_pred_ins = tf.reshape(bbox_pred, [-1, cfgs.CLASS_NUM + 1, 4])

    # only keep a box which score is the bigest
    keep_abox = tf.argmax(cls_prob, axis=1)
    keep_inds = tf.reshape(tf.transpose(tf.stack([tf.cumsum(tf.ones_like(keep_abox)) - 1, keep_abox])),
                           [-1, 2])
    bbox_pred_fliter = tf.reshape(tf.gather_nd(bbox_pred_ins, keep_inds), [-1, 4])

    # decode boxes
    decoded_boxes = encode_and_decode.decode_boxes(encoded_boxes=bbox_pred_fliter,
                                                   reference_boxes=rois,
                                                   scale_factors=cfgs.ROI_SCALE_FACTORS[stage-1])

    return decoded_boxes
