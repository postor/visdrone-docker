import numpy as np
from ensemble_boxes import *


import os

def wbf_fusion(DIR_1,DIR_2,RESULTS_DIR):

    for i in os.listdir(DIR_1):
        boxes_list1 = []
        scores_list1 = []
        labels_list1 = []

        boxes_list2 = []
        scores_list2 = []
        labels_list2 = []

        e_boxes_list = []
        e_scores_list = []
        e_labels_list = []

        with open(os.path.join(DIR_1, i), 'r', encoding='utf-8') as f1:
            img_shape = f1.readlines()[0].strip("\n").split(",")
            img_h = int(img_shape[1])
            img_w = int(img_shape[0])
        with open(os.path.join(DIR_1, i), 'r', encoding='utf-8') as f1:
            for _ in f1.readlines()[1:]:
                _=_.strip("\n").split(",")
                _=list(map(float, _))
                _[-1]=int(_[-1])
                boxes_list1.append(_[:4])
                scores_list1.append(_[4])
                labels_list1.append(_[5])

        with open(os.path.join(DIR_2, i), 'r', encoding='utf-8') as f2:
            for _ in f2.readlines()[1:]:
                _=_.strip("\n").split(",")
                _=list(map(float, _))
                _[-1]=int(_[-1])
                boxes_list2.append(_[:4])
                scores_list2.append(_[4])
                labels_list2.append(_[5])


        e_boxes_list.append(boxes_list1),e_boxes_list.append(boxes_list2)
        e_scores_list.append(scores_list1),e_scores_list.append(scores_list2)
        e_labels_list.append(labels_list1),e_labels_list.append(labels_list2)


        weights=None
        iou_thr = 0.65
        skip_box_thr = 0.001

        boxes, scores, labels = weighted_boxes_fusion(e_boxes_list, e_scores_list, e_labels_list, weights=weights,
                                                      iou_thr=iou_thr, skip_box_thr=skip_box_thr, conf_type='avg')

        # convert normalized boxes to initial boxes
        #
        for box in boxes:
            box[0], box[2] = box[0]*img_w, box[2]*img_w
            box[1], box[3] = box[1]*img_h, box[3]*img_h

        final_boxes = np.transpose(np.stack([boxes[:,0], boxes[:,1], boxes[:,2] - boxes[:,0], boxes[:,3] - boxes[:,1]]))

        with open(os.path.join(RESULTS_DIR, i), "w", encoding="utf-8") as fout:
            for j, box in enumerate(final_boxes):
                    fout.write(str(box[0]))
                    fout.write(',')
                    fout.write(str(box[1]))
                    fout.write(',')
                    fout.write(str(box[2]))
                    fout.write(',')
                    fout.write(str(box[3]))
                    fout.write(',')
                    fout.write(str(scores[j]))
                    fout.write(',')
                    fout.write(str(labels[j]))
                    fout.write(',-1,')
                    fout.write('-1' + '\n')
        print("file name:",i)

if __name__ == '__main__':
    DIR_1 = r'./test_model_ensemble/resnet101_v1d'
    DIR_2 = r'./test_model_ensemble/resnet50_v1d'
    RESULTS_DIR = r'./test_model_ensemble/enemble_ressults'
    wbf_fusion(DIR_1,DIR_2, RESULTS_DIR)

