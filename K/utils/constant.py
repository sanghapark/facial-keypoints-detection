IMAGE_SIZE = 96
COLOR_CHANNEL = 1


FILEPATH_TRAIN = '../data/training.csv'
FILEPATH_TEST  = '../data/test.csv'
ID_LOOKUP_TABLE = '../data/IdLookUpTable.csv'


COLS01 = [
    'left_eye_center_x',            'left_eye_center_y',
    'right_eye_center_x',           'right_eye_center_y',

    'nose_tip_x',                   'nose_tip_y',

    'mouth_center_bottom_lip_x',    'mouth_center_bottom_lip_y',
    'Image'
]
FLIP_INDICES01 = [
    (0, 2),
    (1, 3)
]

# (2155, 23)
COLS02 = [

    'left_eye_inner_corner_x',      'left_eye_inner_corner_y',
    'left_eye_outer_corner_x',      'left_eye_outer_corner_y',
    'right_eye_inner_corner_x',     'right_eye_inner_corner_y',
    'right_eye_outer_corner_x',     'right_eye_outer_corner_y',
    'left_eyebrow_inner_end_x',     'left_eyebrow_inner_end_y',
    'left_eyebrow_outer_end_x',     'left_eyebrow_outer_end_y',
    'right_eyebrow_inner_end_x',    'right_eyebrow_inner_end_y',
    'right_eyebrow_outer_end_x',    'right_eyebrow_outer_end_y',

    'mouth_left_corner_x',          'mouth_left_corner_y',
    'mouth_right_corner_x',         'mouth_right_corner_y',
    'mouth_center_top_lip_x',       'mouth_center_top_lip_y',
    'Image'
]
FLIP_INDICES02 = [
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
    (8, 12),
    (9, 13),
    (10, 14),
    (11, 15),
    (16, 18),
    (17, 19)
]

KEYPOINTS = [
    'left_eye_center_x', 'left_eye_center_y',
    'right_eye_center_x', 'right_eye_center_y',
    'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
    'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
    'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
    'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
    'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
    'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
    'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
    'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
    'nose_tip_x', 'nose_tip_y',
    'mouth_left_corner_x', 'mouth_left_corner_y',
    'mouth_right_corner_x', 'mouth_right_corner_y',
    'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
    'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y'
]

BATCH_SIZE = 128
EPOCHS01 = 1000
EPOCHS02 = 1000
VALIDATION_RATIO = 0.2

ACTIVATION = 'elu'
LAST_ACTIVATION = 'tanh'

FLIP = True
ROTATE = True
CONTRAST = True
PERSPECTIVE_TRANSFORM = True
ELASTIC_TRANSFORM = True

FLIP_RATIO = 0.5
ROTATE_RATIO = 0.5
CONTRAST_RATIO = 0.5
PERSPECTIVE_TRANSFORM_RATIO = 0.5
ELASTIC_TRANSFORM_RATIO = 0.5
