import tensorflow as tf
from core.losses.smoothl1_loss import smooth_l1_loss_rpn, smooth_l1_loss_rcnn
import tensorflow.contrib.slim as slim
from core.config import cfgs


def compute_losses(rpn_box_pred, rpn_bbox_targets, rpn_cls_score, rpn_labels,
                   bbox_pred, bbox_targets, cls_score, labels):
    rpn_cls_loss = 0.0
    rpn_bbox_loss = 0.0
    # rpn
    if rpn_box_pred is not None:
        rpn_bbox_loss = smooth_l1_loss_rpn(rpn_box_pred, rpn_bbox_targets, rpn_labels, sigma=cfgs.RPN_SIGMA_FACTOR)

        label_select = tf.reshape(tf.where(tf.not_equal(rpn_labels, -1)), [-1])
        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, label_select), [-1, 2])
        rpn_labels = tf.reshape(tf.gather(rpn_labels, label_select), [-1])

        rpn_cls_loss = slim.losses.sparse_softmax_cross_entropy(rpn_cls_score, rpn_labels)

        rpn_cls_loss = rpn_cls_loss * cfgs.RPN_CLS_LOSS_WEIGHT
        rpn_bbox_loss = rpn_bbox_loss * cfgs.RPN_BBOX_LOSS_WEIGHT

    rcnn_bbox_loss = smooth_l1_loss_rcnn(bbox_pred, bbox_targets, labels, num_classes=cfgs.CLASS_NUM + 1,
                                         sigma=cfgs.RCNN_SIGMA_FACTOR)
    rcnn_cls_loss = slim.losses.sparse_softmax_cross_entropy(cls_score, labels)
    rcnn_bbox_loss = rcnn_bbox_loss * cfgs.RCNN_BBOX_LOSS_WEIGHT
    rcnn_cls_loss = rcnn_cls_loss * cfgs.RCNN_CLS_LOSS_WEIGHT

    loss_dict = {
        'rpn_cls_loss': rpn_cls_loss,
        'rpn_bbox_loss': rpn_bbox_loss,
        'rcnn_cls_loss': rcnn_cls_loss,
        'rcnn_bbox_loss': rcnn_bbox_loss
    }

    return loss_dict
