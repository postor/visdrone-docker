import tensorflow as tf
import tensorflow.contrib.slim as slim
from core.config import cfgs


def get_pyramid_layer(C_i, P_j, scope):
    with tf.variable_scope(scope):
        level_name = scope.split('_')[1]
        # in visdrone dataset, nearest is better than bilinear
        return tf.image.resize_nearest_neighbor(P_j, size=[tf.shape(C_i)[1], tf.shape(C_i)[2]],
                                                name='up_sample_' + level_name) + \
               slim.conv2d(C_i, num_outputs=256, kernel_size=[1, 1], stride=1, scope='reduce_' + level_name)


def fpn(inputs):
    pyramid_dict = {}
    feature_dict = inputs
    with tf.variable_scope('build_pyramid'):
        with slim.arg_scope([slim.conv2d], weights_regularizer=cfgs.WEIGHT_REGULARIZER,
                            activation_fn=None, normalizer_fn=None):

            P5 = slim.conv2d(feature_dict['C5'],
                             num_outputs=256,
                             kernel_size=[1, 1],
                             stride=1, scope='build_P5')
            if cfgs.GLOBAL_CTX:
                print("×->ADD GLOBAL CTX")
                global_ctx = tf.reduce_mean(feature_dict['C5'], axis=[1, 2], keep_dims=True)
                global_ctx = slim.conv2d(global_ctx, kernel_size=[1, 1], num_outputs=256, stride=1,
                                         activation_fn=None, scope='global_ctx')
                pyramid_dict['P5'] = P5 + global_ctx
            else:
                pyramid_dict['P5'] = P5


            for level in range(4, 1, -1):
                pyramid_dict['P%d' % level] = get_pyramid_layer(C_i=feature_dict["C%d" % level],
                                                                P_j=pyramid_dict["P%d" % (level + 1)],
                                                                scope='build_P%d' % level)

            for level in range(5, 1, -1):
                pyramid_dict['P%d' % level] = slim.conv2d(pyramid_dict['P%d' % level],
                                                          num_outputs=256, kernel_size=[3, 3], padding="SAME",
                                                          stride=1, scope="conv_P%d" % level)

            if "P6" in cfgs.LEVELS:
                P6 = slim.avg_pool2d(pyramid_dict['P5'], kernel_size=[1, 1], stride=2, scope='build_P6')
                pyramid_dict['P6'] = P6

    return [pyramid_dict[level_name] for level_name in cfgs.LEVELS]
