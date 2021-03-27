from core.config import cfgs
from core.box_utils import encode_and_decode
from core.postprocess import boxes_process
import tensorflow as tf

def postprocess_rpn_proposals(rpn_bbox_pred, rpn_cls_prob, img_shape, anchors, is_training):
    '''

    :param rpn_bbox_pred: [-1, 4]
    :param rpn_cls_prob: [-1, 2]
    :param img_shape:
    :param anchors:[-1, 4]
    :param is_training:
    :return:
    '''

    if is_training:
        pre_nms_topN = cfgs.RPN_PRE_NMS_TRAIN
        post_nms_topN = cfgs.RPN_MAXIMUM_PROPOSAL_TARIN

    else:
        pre_nms_topN = cfgs.RPN_PRE_NMS_TEST
        post_nms_topN = cfgs.RPN_MAXIMUM_PROPOSAL_TEST


    nms_thresh = cfgs.RPN_NMS_IOU_THR

    cls_prob = rpn_cls_prob[:, 1]

    # 1. decode boxes
    decode_boxes = encode_and_decode.decode_boxes(encoded_boxes=rpn_bbox_pred,
                                                  reference_boxes=anchors,
                                                  scale_factors=cfgs.ANCHOR_SCALE_FACTORS)

    # 2. clip to img boundaries
    decode_boxes = boxes_process.clip_boxes_to_img_boundaries(decode_boxes=decode_boxes,
                                                            img_shape=img_shape)

    # 3. get top N to NMS
    if pre_nms_topN > 0:
        pre_nms_topN = tf.minimum(pre_nms_topN, tf.shape(decode_boxes)[0], name='avoid_unenough_boxes')
        cls_prob, top_k_indices = tf.nn.top_k(cls_prob, k=pre_nms_topN)
        decode_boxes = tf.gather(decode_boxes, top_k_indices)

    # 4. NMS
    # import time
    # start_time=time.time()
    keep = tf.image.non_max_suppression(
        boxes=decode_boxes,
        scores=cls_prob,
        max_output_size=post_nms_topN,
        iou_threshold=nms_thresh)
    # end_time = time.time()
    # print(end_time - start_time)
    rois = tf.gather(decode_boxes, keep)
    roi_cls_probs = tf.gather(cls_prob, keep)

    return rois, roi_cls_probs

