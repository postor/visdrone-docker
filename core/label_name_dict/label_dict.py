from core.config import cfgs

if cfgs.DATASETS_NAME == 'hkb':

    label_name_dict = {
        'back_ground': 0,
        'Vehicle': 1
    }

elif cfgs.DATASETS_NAME == 'visdrone':
    label_name_dict = {
        'back_ground': 0,
        'ignored regions': 1,
        'pedestrian': 2,
        'people': 3,
        'bicycle': 4,
        'car': 5,
        'van': 6,
        'truck': 7,
        'tricycle': 8,
        'awning-tricycle': 9,
        'bus': 10,
        'motor': 11,
        'others': 12
    }
elif cfgs.DATASETS_NAME == 'pascal':
    label_name_dict = {
        'back_ground': 0,
        'aeroplane': 1,
        'bicycle': 2,
        'bird': 3,
        'boat': 4,
        'bottle': 5,
        'bus': 6,
        'car': 7,
        'cat': 8,
        'chair': 9,
        'cow': 10,
        'diningtable': 11,
        'dog': 12,
        'horse': 13,
        'motorbike': 14,
        'person': 15,
        'pottedplant': 16,
        'sheep': 17,
        'sofa': 18,
        'train': 19,
        'tvmonitor': 20
    }
elif cfgs.DATASETS_NAME.startswith('dota'):
    label_name_dict = {
        'back_ground': 0,
        'roundabout': 1,
        'tennis-court': 2,
        'swimming-pool': 3,
        'storage-tank': 4,
        'soccer-ball-field': 5,
        'small-vehicle': 6,
        'ship': 7,
        'plane': 8,
        'large-vehicle': 9,
        'helicopter': 10,
        'harbor': 11,
        'ground-track-field': 12,
        'bridge': 13,
        'basketball-court': 14,
        'baseball-diamond': 15
    }
