import numpy as np
import tensorflow as tf
class ImgPreprocess():
    def __init__(self,is_training):
        self.is_training=is_training

    def img_resize(self, img_tensor, gtboxes_and_label, target_shortside_len, length_limitation):
        '''
        :param img_tensor:[h, w, c], gtboxes_and_label:[-1, 5].  gtboxes: [xmin, ymin, xmax, ymax]
        :param target_shortside_len:
        :param length_limitation: set max length to avoid OUT OF MEMORY
        :return:
        '''
        def max_length_limitation(length, length_limitation):
            return tf.cond(tf.less(length, length_limitation),
                           true_fn=lambda: length,
                           false_fn=lambda: length_limitation)

        img_h,img_w=tf.shape(img_tensor)[0],tf.shape(img_tensor)[1]

        img_new_h, img_new_w = tf.cond(tf.less(img_h, img_w),
                               true_fn=lambda: (target_shortside_len,
                                                max_length_limitation(target_shortside_len * img_w // img_h,
                                                                      length_limitation)),
                               false_fn=lambda: (
                               max_length_limitation(target_shortside_len * img_h // img_w, length_limitation),
                               target_shortside_len))

        img_tensor = tf.expand_dims(img_tensor, axis=0)
        img_tensor = tf.image.resize_bilinear(img_tensor, [img_new_h, img_new_w])
        if gtboxes_and_label is not None and self.is_training:
            xmin, ymin, xmax, ymax, label = tf.unstack(gtboxes_and_label, axis=1)

            new_xmin, new_ymin = xmin * img_new_w // img_w, ymin * img_new_h // img_h
            new_xmax, new_ymax = xmax * img_new_w // img_w, ymax * img_new_h // img_h
        img_tensor = tf.squeeze(img_tensor, axis=0)

        if gtboxes_and_label is not None and self.is_training:
            return img_tensor, tf.transpose(tf.stack([new_xmin, new_ymin, new_xmax, new_ymax, label], axis=0))
        else:
            return img_tensor

    def random_flip_left_right(self,img_tensor, gtboxes_and_label):

        def flip_left_to_right(img_tensor, gtboxes_and_label):
            h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]

            img_tensor = tf.image.flip_left_right(img_tensor)

            xmin, ymin, xmax, ymax, label = tf.unstack(gtboxes_and_label, axis=1)
            new_xmax = w - xmin
            new_xmin = w - xmax

            return img_tensor, tf.transpose(tf.stack([new_xmin, ymin, new_xmax, ymax, label], axis=0))

        img_tensor, gtboxes_and_label= tf.cond(tf.less(tf.random.uniform(shape=[], minval=0, maxval=1), 0.5),
                                                lambda: flip_left_to_right(img_tensor, gtboxes_and_label),
                                                lambda: (img_tensor, gtboxes_and_label))

        return img_tensor,  gtboxes_and_label

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    import tensorflow as tf
    test=ImgPreprocess(is_training=True)
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])
    output=test.img_resize(img_plac,None,10,100)
    with tf.Session() as sess:
        img_out=sess.run(output, feed_dict={img_plac:np.random.uniform(low=0,high=255,size=(12,12,3))})
        print(img_out)


