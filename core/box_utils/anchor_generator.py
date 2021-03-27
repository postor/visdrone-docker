import tensorflow as tf
import numpy as np


def make_anchors(
        base_anchor_scale,
        anchor_ratio,
        feature_map_h,
        feature_map_w,
        stride):
    '''
    :param base_anchor_scale:[]
    :param anchor_ratio:[0.5,1.0,2.0]
    :param feature_map_h:
    :param feature_map_w:
    :param stride:
    :return: [xmin,ymin,xmax,ymax]
    '''
    base_anchor_scale = np.array(base_anchor_scale, dtype=np.float32)
    anchor_ratio = np.array(anchor_ratio, dtype=np.float32)
    stride = np.array(stride,dtype=np.float32)
    h = tf.multiply(tf.range(feature_map_h),stride)
    w = tf.multiply(tf.range(feature_map_w),stride)
    center_x, center_y = tf.meshgrid(w, h)
    center_x, center_y = tf.cast(center_x, tf.float32), tf.cast(center_y, tf.float32)

    scale, ratio = tf.meshgrid(base_anchor_scale, anchor_ratio)
    scale, ratio = tf.keras.layers.Flatten()(
        scale), tf.keras.layers.Flatten()(ratio)
    w, h = scale / tf.sqrt(ratio), scale * tf.sqrt(ratio)
    h, w = tf.cast(h, tf.float32), tf.cast(w, tf.float32)
    w, center_x = tf.meshgrid(w, center_x)
    h, center_y = tf.meshgrid(h, center_y)

    anchor_center = tf.stack([center_x, center_y], axis=2)
    anchor_center = tf.reshape(anchor_center, (-1, 2))
    anchor_size = tf.stack([w, h], axis=2)
    anchor_size = tf.reshape(anchor_size, (-1, 2))
    anchors = tf.concat([anchor_center - tf.multiply(0.5, anchor_size),
                         anchor_center + tf.multiply(0.5, anchor_size)], axis=1)
    return anchors

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as path

    scale = [8]
    ratio = np.array([0.5, 1.0, 2.0])
    f_h = 16
    f_w = 16
    anchors = make_anchors(scale, ratio, f_h, f_w, 16)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        anchor_result = sess.run(anchors)
        print (anchor_result.shape)
    anchors = np.array(anchor_result, dtype=np.float32)  # convert to numpy type

    print(anchors.shape)
    img = np.ones((128, 128, 3))
    plt.imshow(img)
    AXS = plt.gca()
    for i in range(anchors.shape[0]):
        box = anchors[i]
        m = path.Rectangle(
            (box[0],
             box[1]),
            box[2] - box[0],
            box[3] - box[1],
            edgecolor="r",
            facecolor="none")
        AXS.add_patch(m)
    plt.show()
