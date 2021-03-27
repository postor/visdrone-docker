import tensorflow as tf
from tensorflow.contrib import slim
from core.config import cfgs
import os
from data.load_datasets import DatasetLoader
from data.image_preprocess import ImgPreprocess
from models.meta_architectures import BuildFrameworks
from core.summary_bbox import show_box_in_tensor
import numpy as np
import time

def train():
    # initialize
    net = BuildFrameworks(True)
    img_preprocess = ImgPreprocess(is_training=True)
    dataset_loader = DatasetLoader()
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])
    gtbox_plac = tf.placeholder(dtype=tf.int32, shape=[None, 5])
    #multi-scale training:
    if cfgs.MULTI_SCALSE_TRAINING:
        shortside_len_list = tf.constant(cfgs.SHORT_SIZE_LIMITATION_LIST)
        shortside_len = tf.random_shuffle(shortside_len_list)[0]
    else:
        shortside_len = cfgs.SHORT_SIZE_LIMITATION
    img, gtboxes_and_label = img_preprocess.img_resize(img_tensor=img_plac,
                                                       gtboxes_and_label=gtbox_plac,
                                                       target_shortside_len=shortside_len,
                                                       length_limitation=cfgs.MAX_LENGTH)

    img, gtboxes_and_label = img_preprocess.random_flip_left_right(img_tensor=img,
                                                                   gtboxes_and_label=gtboxes_and_label)

    if cfgs.PRETRAIN_BACKBONE_NAME in ['resnet101_v1d', 'resnet50_v1d']:
        img_tensor = img / 255 - tf.constant([[cfgs.PIXEL_MEAN_]])
        img_tensor = img_tensor / tf.constant([cfgs.PIXEL_STD])
    else:
        img_tensor = img - tf.constant([[cfgs.PIXEL_MEAN]])

    img_tensor = tf.expand_dims(img_tensor, axis=0)
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=cfgs.WEIGHT_REGULARIZER,
                        biases_initializer=tf.constant_initializer(0.0)):
        loss_dict, det_dict, anchors_summary_dict, rpn_rois_summary_dict = net.build_framworks(
            input_img_batch=img_tensor, gtboxes_batch=gtboxes_and_label)

    # loss
    if cfgs.WEIGHT_REGULARIZER:
        weight_decay_loss = tf.add_n(slim.losses.get_regularization_losses())
    else:
        weight_decay_loss = 0
    rpn_cls_loss, rpn_bbox_loss, rcnn_cls_loss, rcnn_bbox_loss = loss_dict["rpn_cls_loss"], loss_dict[
        "rpn_bbox_loss"], loss_dict["rcnn_cls_loss"], loss_dict["rcnn_bbox_loss"]
    total_loss = rpn_cls_loss + rpn_bbox_loss + rcnn_cls_loss + rcnn_bbox_loss + weight_decay_loss

    # detection result dict
    final_boxes, final_scores, final_category = det_dict["final_boxes"], det_dict["final_scores"], det_dict[
        "final_category"]

    # summary
    if cfgs.USE_SUMMARY:
        gtboxes_in_img = show_box_in_tensor.draw_boxes_with_categories(img_batch=img_tensor,
                                                                       boxes=gtboxes_and_label[:, :-1],
                                                                       labels=gtboxes_and_label[:, -1])
        detections_in_img = show_box_in_tensor.draw_boxes_with_categories_and_scores(img_batch=img_tensor,
                                                                                     boxes=final_boxes,
                                                                                     labels=final_category,
                                                                                     scores=final_scores)

        # writing summary data
        tf.summary.scalar("total_loss", total_loss)
        tf.summary.scalar("rpn_cls_loss", rpn_cls_loss)
        tf.summary.scalar("rpn_bbox_loss", rpn_bbox_loss)
        tf.summary.scalar("rcnn_cls_loss", rcnn_cls_loss)
        tf.summary.scalar("rcnn_bbox_loss", rcnn_bbox_loss)
        tf.summary.image('Compare/final_detection', detections_in_img)
        tf.summary.image('Compare/gtboxes', gtboxes_in_img)
        tf.summary.image("anchors/pos_in_img", anchors_summary_dict["pos_in_img"])
        tf.summary.image("anchors/neg_in_img", anchors_summary_dict["neg_in_img"])
        tf.summary.image("rois/rpn_rois", rpn_rois_summary_dict["rpn_rois"])
    # tf.summary.image("rois/pos_in_img", rois_summary_dict["pos_in_img"])
    # tf.summary.image("rois/neg_in_img", rois_summary_dict["neg_in_img"])

    global_step = tf.train.get_or_create_global_step()
    lr = tf.train.piecewise_constant(global_step,
                                     boundaries=[np.int64(cfgs.DECAY_STEP[0]), np.int64(cfgs.DECAY_STEP[1])],
                                     values=[cfgs.LR, cfgs.LR / 10., cfgs.LR / 100.])
    if cfgs.USE_SUMMARY:
        tf.summary.scalar('lr', lr)
    optimizer = tf.train.MomentumOptimizer(lr, momentum=cfgs.MOMENTUM)

    # optimizer=tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
    gradients = optimizer.compute_gradients(total_loss)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients,
                                         global_step=global_step)
    summary_op = tf.summary.merge_all()
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    restorer, restore_ckpt = net.get_restorer()
    saver = tf.train.Saver(max_to_keep=25)
    with tf.Session() as sess:
        sess.run(init_op)
        restorer.restore(sess, restore_ckpt)
        summary_path = cfgs.SUMMARY_DIR + cfgs.DATASETS_NAME + "/" + cfgs.PRETRAIN_BACKBONE_NAME
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        if cfgs.USE_SUMMARY:
            summary_writer = tf.summary.FileWriter(summary_path, graph=sess.graph)
        start_global_step_ = sess.run(global_step)

        assert start_global_step_ < cfgs.MAX_ITERATION
        for step in range(start_global_step_ + 1, cfgs.MAX_ITERATION + 1):
            imgid, img, gtbox = dataset_loader.read_next_img(step=step, start_global_step=start_global_step_)
            sess.run(train_op, feed_dict={img_plac: img, gtbox_plac: gtbox})

            if step % 10 == 0 or step == 1:
                start_time = time.time()
                rpn_bbox_loss_, rpn_cls_loss_, rcnn_bbox_loss_, rcnn_cls_loss_, total_loss_, lr_ = \
                    sess.run([rpn_bbox_loss, rpn_cls_loss, rcnn_bbox_loss,
                              rcnn_cls_loss, total_loss, lr], feed_dict={img_plac: img, gtbox_plac: gtbox})
                end_time = time.time()
                print("step:", step, "|", "lr:", lr_, "img_name:", imgid)
                print("total_loss:", total_loss_)
                print("rpn_cls_loss:", rpn_cls_loss_, "rpn_bbox_loss:", rpn_bbox_loss_)
                print("rcnn_cls_loss:", rcnn_cls_loss_, "rcnn_bbox_loss:", rcnn_bbox_loss_)
                print(f"training speed per step:{end_time - start_time}s")
                print("-" * 100)

            if step % cfgs.SUMMARY_STEP == 0 and cfgs.USE_SUMMARY:
                summary_str = sess.run(summary_op, feed_dict={img_plac: img, gtbox_plac: gtbox})
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if step % cfgs.SAVED_STEP == 0 or step == cfgs.MAX_ITERATION:
                if not os.path.exists(cfgs.SAVED_WEIGHTS_DIR):
                    os.mkdir(cfgs.SAVED_WEIGHTS_DIR)
                save_ckpt = os.path.join(cfgs.SAVED_WEIGHTS_DIR, 'visdrone_' + cfgs.PRETRAIN_BACKBONE_NAME+ '_'+ str(step) + '_model.ckpt')
                saver.save(sess, save_ckpt)
                print("trained weights has been saved in the directory:", cfgs.SAVED_WEIGHTS_DIR)


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    train()
