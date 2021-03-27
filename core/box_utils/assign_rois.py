import tensorflow as tf
from core.config import cfgs
def assign_levels(all_rois, labels=None, bbox_targets=None, is_training=False):
    '''
    :param all_rois:
    :param labels:
    :param bbox_targets:
    :return:
    '''
    with tf.name_scope('assign_levels'):
        xmin, ymin, xmax, ymax = tf.unstack(all_rois, axis=1)
        h = tf.maximum(0., ymax - ymin)
        w = tf.maximum(0., xmax - xmin)

        levels = tf.floor(4. + tf.math.log(tf.sqrt(w * h + 1e-8) / 224.0) / tf.math.log(2.))  # 4 + log_2(***)
        min_level = int(cfgs.LEVELS[0][-1])
        max_level = min(5, int(cfgs.LEVELS[-1][-1]))
        levels = tf.maximum(levels, tf.ones_like(levels) * min_level)  # level minimum is 2
        levels = tf.minimum(levels, tf.ones_like(levels) * max_level)  # level maximum is 5

        levels = tf.stop_gradient(tf.reshape(levels, [-1]))

        def get_rois(levels, level_i, rois, labels, bbox_targets):

            level_i_indices = tf.reshape(tf.where(tf.equal(levels, level_i)), [-1])
            level_i_rois = tf.gather(rois, level_i_indices)

            if is_training:
                level_i_rois = tf.stop_gradient(level_i_rois)
                level_i_labels = tf.gather(labels, level_i_indices)
                level_i_targets = tf.gather(bbox_targets, level_i_indices)

                return level_i_rois, level_i_labels, level_i_targets
            else:
                return level_i_rois, None, None

        rois_list = []
        labels_list = []
        targets_list = []
        for i in range(min_level, max_level + 1):
            P_i_rois, P_i_labels, P_i_targets = get_rois(levels, level_i=i, rois=all_rois,
                                                         labels=labels,
                                                         bbox_targets=bbox_targets)
            rois_list.append(P_i_rois)
            labels_list.append(P_i_labels)
            targets_list.append(P_i_targets)

        if is_training:
            all_labels = tf.concat(labels_list, axis=0)
            all_targets = tf.concat(targets_list, axis=0)
            return rois_list, all_labels, all_targets
        else:
            return rois_list