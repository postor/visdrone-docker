import tensorflow as tf
def clip_boxes_to_img_boundaries(decode_boxes, img_shape):

    xmin = decode_boxes[:, 0]
    ymin = decode_boxes[:, 1]
    xmax = decode_boxes[:, 2]
    ymax = decode_boxes[:, 3]
    img_h, img_w = img_shape[1], img_shape[2]

    img_h, img_w = tf.cast(img_h, tf.float32), tf.cast(img_w, tf.float32)

    xmin = tf.maximum(tf.minimum(xmin, img_w-1.), 0.)
    ymin = tf.maximum(tf.minimum(ymin, img_h-1.), 0.)

    xmax = tf.maximum(tf.minimum(xmax, img_w-1.), 0.)
    ymax = tf.maximum(tf.minimum(ymax, img_h-1.), 0.)

    return tf.transpose(tf.stack([xmin, ymin, xmax, ymax]))

