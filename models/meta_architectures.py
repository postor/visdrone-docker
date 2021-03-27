import tensorflow as tf
import os
from tensorflow.contrib import slim
from core.config import cfgs
from models.backbone.resnet_vd import resnet_vd
from models.rpn import RPN
from models.fpn import fpn
from models.fast_rcnn import FastRCNN
from core.box_utils.anchor_generator import make_anchors
from core.postprocess.postprocess_rpn import postprocess_rpn_proposals
from core.box_utils.assign_rois import assign_levels
from core.postprocess.postprocess_fastrcnn import postprocess_fastrcnn, postprocess_cascadercnn
from core.losses.losses import compute_losses
from core.box_utils.anchor_target_assigner import anchor_target_layer
from core.box_utils.proposal_target_assigner import proposal_target_layer
from core.summary_bbox import show_box_in_tensor


class BuildFrameworks(tf.keras.Model):
    def __init__(self, is_training):
        super(BuildFrameworks, self).__init__()
        self.is_training = is_training
        self.anchors_summary_dict = {}
        self.rpn_rois_summary_dict = {}
        self.build_rpn = RPN()
        self.build_fast_rcnn = FastRCNN()

    def build_framworks(self, input_img_batch, gtboxes_batch=None):
        img_shape = tf.shape(input_img_batch)
        if self.is_training:
            gtboxes_batch = tf.reshape(gtboxes_batch, [-1, 5])
            gtboxes_batch = tf.cast(gtboxes_batch, tf.float32)
        if cfgs.PRETRAIN_BACKBONE_NAME in ['resnet101_v1d', 'resnet50_v1d']:
            features_from_backbone = resnet_vd(
                input_img_batch,
                cfgs.PRETRAIN_BACKBONE_NAME,
                is_training=self.is_training)
        # elif cfgs.PRETRAIN_BACKBONE_NAME in ["", ""]:

        else:
            raise AssertionError()
        # fpn
        pyramid_layers_list = fpn(features_from_backbone)
        # rpn
        fpn_cls_score = []
        fpn_box_pred = []
        for level_name, p in zip(cfgs.LEVELS, pyramid_layers_list):
            rpn_box_pred, rpn_cls_prob = self.build_rpn.rpn(
                p, level_name, self.is_training)
            fpn_box_pred.append(rpn_box_pred)
            fpn_cls_score.append(rpn_cls_prob)

        rpn_cls_score = tf.concat(fpn_cls_score, axis=0)
        rpn_box_pred = tf.concat(fpn_box_pred, axis=0)
        rpn_cls_prob = slim.softmax(rpn_cls_score, scope='rpn_cls_prob_sotmax')

        # generate anchors for pyramid layers
        all_anchors = []
        for i in range(len(cfgs.LEVELS)):
            fmap_h = tf.shape(pyramid_layers_list[i])[1]
            fmap_w = tf.shape(pyramid_layers_list[i])[2]
            anchors = make_anchors(
                cfgs.BASE_ANCHOR_SIZE_LIST[i],
                cfgs.ANCHOR_RATIOS,
                fmap_h,
                fmap_w,
                stride=cfgs.ANCHOR_STRIDE_LIST[i])
            all_anchors.append(anchors)
        all_anchors = tf.concat(all_anchors, axis=0)

        # postprocess rpn
        rois, roi_cls_probs = postprocess_rpn_proposals(
            rpn_box_pred, rpn_cls_prob, img_shape, all_anchors, is_training=self.is_training)

        if self.is_training:
            score_gre_05 = tf.reshape(
                tf.where(tf.greater_equal(roi_cls_probs, 0.5)), [-1])
            score_gre_05_rois = tf.gather(rois, score_gre_05)
            score_gre_05_score = tf.gather(roi_cls_probs, score_gre_05)
            score_gre_05_in_img = show_box_in_tensor.draw_boxes_with_scores(
                img_batch=input_img_batch, boxes=score_gre_05_rois, scores=score_gre_05_score)

            self.rpn_rois_summary_dict['rpn_rois'] = score_gre_05_in_img

        # generate label
        if self.is_training:
            rpn_labels, rpn_bbox_targets = tf.py_func(
                anchor_target_layer, [
                    gtboxes_batch, img_shape, all_anchors], [
                    tf.float32, tf.float32])

            rpn_bbox_targets = tf.reshape(rpn_bbox_targets, [-1, 4])
            rpn_labels = tf.cast(rpn_labels, tf.int32)
            rpn_labels = tf.reshape(rpn_labels, [-1])
            if cfgs.USE_SUMMARY:
                self.anchors_summary_dict = self.add_anchor_img_smry(
                    input_img_batch, all_anchors, rpn_labels)  # summary

        total_loss_dict = {}
        stage_threshold = cfgs.CASCADE_SAMPLE_IOU_THR
        stage = [1, 2, 3]
        # start_time=time.time()
        for i, j in zip(stage[:-1], stage_threshold[:-1]):
            if self.is_training:
                loss_dict, rois, _, _ = self.build_fast_rcnn.cascade_rcnn(self.is_training,
                                                                                         rois,
                                                                                         gtboxes_batch,
                                                                                         input_img_batch,
                                                                                         pyramid_layers_list,
                                                                                         img_shape,
                                                                                         j,
                                                                                         rpn_box_pred,
                                                                                         rpn_bbox_targets,
                                                                                         rpn_cls_score,
                                                                                         rpn_labels,
                                                                                         stage=i)
                for k in loss_dict.keys():
                    if k not in total_loss_dict.keys():
                        total_loss_dict[k] = loss_dict[k]
                    else:
                        total_loss_dict[k] += loss_dict[k]
            else:
                rois, _, _ = self.build_fast_rcnn.cascade_rcnn(self.is_training,
                                                                       rois,
                                                                       gtboxes_batch,
                                                                       input_img_batch,
                                                                       pyramid_layers_list,
                                                                       img_shape,
                                                                       j,
                                                                       rpn_box_pred,
                                                                       None,
                                                                       rpn_cls_score,
                                                                       None,
                                                                       stage=i)
        if self.is_training:
            loss_dict, rois, bbox_pred, cls_prob = self.build_fast_rcnn.cascade_rcnn(self.is_training, rois,
                                                                                     gtboxes_batch,
                                                                                     input_img_batch,
                                                                                     pyramid_layers_list,
                                                                                     img_shape,
                                                                                     stage_threshold[-1],
                                                                                     rpn_box_pred,
                                                                                     rpn_bbox_targets,
                                                                                     rpn_cls_score,
                                                                                     rpn_labels,
                                                                                     stage=stage[-1])
            for k in loss_dict.keys():
                if k not in total_loss_dict.keys():
                    total_loss_dict[k] = loss_dict[k]
                else:
                    total_loss_dict[k] += loss_dict[k]

            det_dict = postprocess_fastrcnn(rois, bbox_pred, cls_prob, img_shape, self.is_training)
            # end_time = time.time()
            # # # time_dict["postprocess cascade"] = end_time - start_time
            # print(time_dict)
            return total_loss_dict, det_dict, self.anchors_summary_dict, self.rpn_rois_summary_dict

        else:
        #good result: 0.8 * cls_prob_stage3 + 0.1 * cls_prob_stage2 + 0.1 * cls_prob_stage1
            rois, bbox_pred, cls_prob_stage3 = self.build_fast_rcnn.cascade_rcnn(self.is_training, rois, gtboxes_batch,
                                                                                 input_img_batch,
                                                                                 pyramid_layers_list,
                                                                                 img_shape, 0,
                                                                                 rpn_box_pred, None,
                                                                                 rpn_cls_score, None,
                                                                                 stage=3)
            cascade_rois_ = postprocess_cascadercnn(rois=rois,
                                                    bbox_pred=bbox_pred,
                                                    cls_prob=cls_prob_stage3,
                                                    stage=3)
            with slim.arg_scope([slim.conv2d, slim.fully_connected], reuse=True):
                _, _, cls_prob_stage1 = self.build_fast_rcnn.cascade_rcnn(self.is_training, cascade_rois_,
                                                                          gtboxes_batch,
                                                                          input_img_batch,
                                                                          pyramid_layers_list,
                                                                          img_shape, 0,
                                                                          rpn_box_pred, None,
                                                                          rpn_cls_score, None,
                                                                          stage=1)

                _, _, cls_prob_stage2 = self.build_fast_rcnn.cascade_rcnn(self.is_training, cascade_rois_,
                                                                          gtboxes_batch,
                                                                          input_img_batch,
                                                                          pyramid_layers_list,
                                                                          img_shape, 0,
                                                                          rpn_box_pred, None,
                                                                          rpn_cls_score, None,
                                                                          stage=2)
            # better accuracy
            cls_prob = 0.8 * cls_prob_stage3 + 0.1 * cls_prob_stage2 + 0.1 * cls_prob_stage1

            det_dict = postprocess_fastrcnn(rois=rois,
                                            bbox_ppred=bbox_pred,
                                            scores=cls_prob,
                                            img_shape=img_shape, is_training=self.is_training)

            return det_dict

    def add_anchor_img_smry(self, img, anchors, labels):

        positive_anchor_indices = tf.reshape(
            tf.where(tf.greater_equal(labels, 1)), [-1])
        negative_anchor_indices = tf.reshape(
            tf.where(tf.equal(labels, 0)), [-1])

        positive_anchor = tf.gather(anchors, positive_anchor_indices)
        negative_anchor = tf.gather(anchors, negative_anchor_indices)

        pos_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                        boxes=positive_anchor)
        neg_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                        boxes=negative_anchor)
        self.anchors_summary_dict['pos_in_img'] = pos_in_img
        self.anchors_summary_dict['neg_in_img'] = neg_in_img

        return self.anchors_summary_dict

    def get_restorer(self):
        checkpoint_path = tf.train.latest_checkpoint(os.path.join(cfgs.SAVED_WEIGHTS_DIR))
        # checkpoint_path = r'./saved_weights/多尺度训练/r101/visdrone_resnet101_v1d_87000_model.ckpt'
        if checkpoint_path is not None:
            restorer = tf.train.Saver()
            print("loading:", checkpoint_path)
        else:
            checkpoint_path = cfgs.PRETRAIN_BACKBONE_WEIGHTS
            print("loading pretrained weights, path is :", checkpoint_path)

            model_variables = slim.get_model_variables()

            def name_in_ckpt_rpn(var):
                return var.op.name

            nameInCkpt_Var_dict = {}
            for var in model_variables:
                if var.name.startswith(cfgs.PRETRAIN_BACKBONE_NAME):
                    var_name_in_ckpt = name_in_ckpt_rpn(var)
                    nameInCkpt_Var_dict[var_name_in_ckpt] = var
            restore_variables = nameInCkpt_Var_dict
            restorer = tf.train.Saver(restore_variables)

        return restorer, checkpoint_path
