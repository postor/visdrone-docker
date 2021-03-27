import numpy as np
import cv2
from core.config import cfgs
import xml.etree.cElementTree as ET
from core.label_name_dict.label_dict import label_name_dict
import os

img_list = os.listdir(
    os.path.join(
        cfgs.TRAIN_DATAETS_DIR,
        "JPEGImages"))
total_imgs = len(img_list)


class DatasetLoader():

    def read_xml_gtbox_and_label(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        img_width = None
        img_height = None
        box_list = []
        for child_of_root in root:
            if child_of_root.tag == 'size':
                for child_item in child_of_root:
                    if child_item.tag == 'width':
                        img_width = int(child_item.text)
                    if child_item.tag == 'height':
                        img_height = int(child_item.text)

            if child_of_root.tag == 'object':
                label = None
                for child_item in child_of_root:
                    if child_item.tag == 'name':
                        label = label_name_dict[child_item.text]
                    if child_item.tag == 'bndbox':
                        tmp_box = []
                        for node in child_item:
                            tmp_box.append(int(node.text))
                        assert label is not None, 'label is none, error'
                        tmp_box.append(label)
                        box_list.append(tmp_box)

        gtbox_label = np.array(box_list, dtype=np.int32)

        return img_height, img_width, gtbox_label

    def read_next_img(self, step, start_global_step):

        if step % total_imgs == 0 or step == start_global_step + 1:
            np.random.shuffle(img_list)
        img_name = img_list[step % total_imgs]

        img = cv2.imread(
            os.path.join(
                cfgs.TRAIN_DATAETS_DIR,
                "JPEGImages",
                img_name))

        img_height, img_width, gtbox_label = self.read_xml_gtbox_and_label(os.path.join(
            cfgs.TRAIN_DATAETS_DIR, 'Annotations', img_name.replace('.jpg', '.xml')))

        gtbox_and_label_list = np.array(gtbox_label, dtype=np.int32)
        if gtbox_and_label_list.shape[0] == 0:
            return self.read_next_img(step + 1, start_global_step)
        else:
            return img_name, img[:, :, ::-1], gtbox_and_label_list


if __name__ == '__main__':
    test = DatasetLoader()
    xml_list = os.listdir(os.path.join(cfgs.TRAIN_DATAETS_DIR, "Annotations"))
    xml_test_file = os.path.join(
        cfgs.TRAIN_DATAETS_DIR,
        "Annotations",
        xml_list[0])
    test.read_xml_gtbox_and_label(xml_test_file)
    imgid, img, gtbox = test.read_next_img(500,10)
    green = (0, 255, 0)
    cv2.line(img[:,:,::-1], (0, 0), (300, 300), green, thickness=4)
    print(gtbox)
    cv2.imshow("img", img[:, :, ::-1])
    # cv2.imshow("img", img[:,:,::-1])
    cv2.waitKey(0)
