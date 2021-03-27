import tensorflow as tf
from tensorflow.contrib import slim as slim
from core.config import cfgs


class RPN():
    def rpn(self, input, level_name, is_training):
        with slim.arg_scope([slim.conv2d], weights_regularizer=cfgs.WEIGHT_REGULARIZER):
            reuse_flag = None if level_name == cfgs.LEVELS[0] else True
            shared_rpn_layer = slim.conv2d(
                input, 512, [3, 3],
                trainable=is_training, weights_initializer=cfgs.INITIALIZER, padding="SAME",
                activation_fn=tf.nn.relu,
                scope='rpn_conv/3x3',
                reuse=reuse_flag)

            rpn_cls_score = slim.conv2d(shared_rpn_layer, 2 * len(cfgs.ANCHOR_RATIOS), [1, 1], stride=1,
                                        trainable=is_training, weights_initializer=cfgs.INITIALIZER,
                                        activation_fn=None, padding="VALID",
                                        scope='rpn_cls_score',
                                        reuse=reuse_flag)
            rpn_box_pred = slim.conv2d(shared_rpn_layer, 4 * len(cfgs.ANCHOR_RATIOS), [1, 1], stride=1,
                                       trainable=is_training, weights_initializer=cfgs.BBOX_INITIALIZER,
                                       activation_fn=None, padding="VALID",
                                       scope='rpn_bbox_pred',
                                       reuse=reuse_flag)

            rpn_box_pred = tf.reshape(rpn_box_pred, [-1, 4])
            rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])
        return rpn_box_pred, rpn_cls_score
