import tensorflow as tf
from tensorflow.contrib import slim
from core.config import cfgs
from core.box_utils.assign_rois import assign_levels
from core.losses.losses import compute_losses
from core.box_utils.proposal_target_assigner import proposal_target_layer
from core.postprocess.postprocess_fastrcnn import postprocess_cascadercnn


class FastRCNN():
    def fast_rcnn(self, input, rois_list, img_shape, is_training):
        '''
        :param input: feature_map
        :param rois_list:
        :param img_shape:
        :return:
        '''
        img_h, img_w = tf.cast(img_shape[1], tf.float32), tf.cast(img_shape[2], tf.float32)
        pooled_features_list = []
        for p, rois in zip(input[:-1], rois_list):
            N = tf.shape(rois)[0]
            x1, y1, x2, y2 = tf.unstack(rois, axis=1)

            normalized_x1 = x1 / img_w
            normalized_x2 = x2 / img_w
            normalized_y1 = y1 / img_h
            normalized_y2 = y2 / img_h

            # build_roi_align
            normalized_rois = tf.transpose(tf.stack([normalized_y1, normalized_x1, normalized_y2, normalized_x2]))
            normalized_rois = tf.stop_gradient(normalized_rois)

            cropped_roi_features = tf.image.crop_and_resize(p, normalized_rois,
                                                            tf.zeros(shape=[N, ], dtype=tf.int32),
                                                            crop_size=(14, 14), method="bilinear")

            pooled_roi_features = slim.max_pool2d(cropped_roi_features, (2, 2), stride=2)
            pooled_features_list.append(pooled_roi_features)

        pooled_features = tf.concat(pooled_features_list, axis=0)
        features_flatten = slim.flatten(pooled_features, scope="flatten_features")
        with slim.arg_scope([slim.fully_connected], weights_regularizer=cfgs.WEIGHT_REGULARIZER):

            fc1 = slim.fully_connected(features_flatten, 1024, scope='fc1')
            fc2 = slim.fully_connected(fc1, 1024, scope='fc2')

            cls_score = slim.fully_connected(fc2,
                                             cfgs.CLASS_NUM + 1,
                                             weights_initializer=cfgs.INITIALIZER,
                                             activation_fn=None,
                                             trainable=is_training,
                                             scope='cls_fc')
            if cfgs.EXTRA_CONV_FOR_REG > 0:
                bbox_input_feat = pooled_features
                for i in range(cfgs.EXTRA_CONV_FOR_REG):
                    bbox_input_feat = slim.conv2d(bbox_input_feat, num_outputs=256, kernel_size=[3, 3], stride=1,
                                                  padding="SAME", scope='extra_conv%d' % i)
                extra_conv_flatten = slim.flatten(bbox_input_feat, scope='bbox_feat_flatten')

                bbox_pred = slim.fully_connected(extra_conv_flatten,
                                                 (cfgs.CLASS_NUM + 1) * 4,
                                                 activation_fn=None,
                                                 trainable=is_training,
                                                 weights_initializer=cfgs.BBOX_INITIALIZER,
                                                 scope='reg_fc')
                print("×-> Use extra conv layers for boxes regression:", cfgs.EXTRA_CONV_FOR_REG)
            else:
                bbox_pred = slim.fully_connected(fc2,
                                                 (cfgs.CLASS_NUM + 1) * 4,
                                                 activation_fn=None,
                                                 trainable=is_training,
                                                 weights_initializer=cfgs.BBOX_INITIALIZER,
                                                 scope='reg_fc')
        cls_score = tf.reshape(cls_score, [-1, cfgs.CLASS_NUM + 1])
        bbox_pred = tf.reshape(bbox_pred, [-1, 4 * (cfgs.CLASS_NUM + 1)])

        return bbox_pred, cls_score

    def cascade_rcnn(self, is_training, rois, gtboxes_batch, input_img_batch, pyramid_layers_list,
                     img_shape, stage_threshold, rpn_box_pred, rpn_bbox_targets,
                     rpn_cls_score, rpn_labels, stage=None):
        '''
        :param rois:
        :param gtboxes_batch:
        :param input_img_batch:
        :param pyramid_layers_list:
        :param img_shape:
        :param stage_threshold: 0.5,0.6,0.7
        :param rpn_box_pred:
        :param rpn_bbox_targets:
        :param rpn_cls_score:
        :param rpn_labels:
        :param stage: 1,2,3
        :return:
        '''
        with tf.variable_scope(f'rois/stage_{stage}'):
            if is_training:
                # import time
                # s=time.time()
                rois, labels, bbox_targets = tf.py_func(
                    proposal_target_layer, [
                        rois, gtboxes_batch,stage_threshold,stage], [
                        tf.float32, tf.float32, tf.float32])
                rois = tf.reshape(rois, [-1, 4])
                labels = tf.cast(labels, tf.int32)
                labels = tf.reshape(labels, [-1])
                if cfgs.USE_SUMMARY:
                    self.add_roi_batch_img_smry(input_img_batch, rois, labels)  # summary
                bbox_targets = tf.reshape(bbox_targets, [-1, 4 * (cfgs.CLASS_NUM + 1)])
                rois_list, labels, bbox_targets = assign_levels(all_rois=rois,
                                                                labels=labels,
                                                                bbox_targets=bbox_targets,
                                                                is_training=is_training)
            else:
                rois_list = assign_levels(all_rois=rois)
            # fast rcnn
            bbox_pred, cls_score = self.fast_rcnn(pyramid_layers_list, rois_list, img_shape, is_training)
            cls_prob = slim.softmax(cls_score, scope="cls_prob_softmax")
            rois = tf.concat(rois_list, axis=0)
            # postprocess cascade rcnn
            if stage != 3:
                rois = postprocess_cascadercnn(rois=rois,
                                               bbox_pred=bbox_pred,
                                               cls_prob=cls_prob,
                                               stage=stage)
                # loss
                if is_training:
                    loss_dict = compute_losses(None, None, None, None, bbox_pred, bbox_targets, cls_score, labels)
                    return loss_dict, rois, bbox_pred, cls_prob

                else:
                    return rois, bbox_pred, cls_prob
            else:
                if is_training:
                    loss_dict = compute_losses(rpn_box_pred, rpn_bbox_targets, rpn_cls_score, rpn_labels, bbox_pred,
                                               bbox_targets,
                                               cls_score, labels)

                    return loss_dict, rois, bbox_pred, cls_prob
                else:
                    return rois, bbox_pred, cls_prob


    def add_roi_batch_img_smry(self, img, rois, labels):
        from core.summary_bbox import show_box_in_tensor
        positive_roi_indices = tf.reshape(
            tf.where(tf.greater_equal(labels, 1)), [-1])

        negative_roi_indices = tf.reshape(tf.where(tf.equal(labels, 0)), [-1])

        pos_roi = tf.gather(rois, positive_roi_indices)
        neg_roi = tf.gather(rois, negative_roi_indices)

        pos_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                        boxes=pos_roi)
        neg_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                        boxes=neg_roi)
        tf.summary.image('pos_in_img', pos_in_img)
        tf.summary.image('neg_in_img', neg_in_img)

