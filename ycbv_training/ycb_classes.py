"""
Maps YCB-Video's 21 standard object classes to our registry vocabulary.
Class IDs and names below match the well-documented, standard YCB-Video
/ PoseCNN ordering (cls_indexes in meta.mat correspond to these IDs).

Only objects with clear shape-category analogs to our existing
vocabulary are mapped; the rest are explicitly excluded (mapped to
None) rather than force-fit into a wrong category -- an honest scoping
decision, not an oversight. If your copy of the dataset uses a
different class list/order (some mirrors vary slightly), print one
frame's meta['cls_indexes'] alongside classes.txt (included in most
YCB-Video distributions) to confirm before trusting this mapping.
"""

YCB_CLASS_NAMES = {
    1: '002_master_chef_can', 2: '003_cracker_box', 3: '004_sugar_box',
    4: '005_tomato_soup_can', 5: '006_mustard_bottle', 6: '007_tuna_fish_can',
    7: '008_pudding_box', 8: '009_gelatin_box', 9: '010_potted_meat_can',
    10: '011_banana', 11: '019_pitcher_base', 12: '021_bleach_cleanser',
    13: '024_bowl', 14: '025_mug', 15: '035_power_drill', 16: '036_wood_block',
    17: '037_scissors', 18: '040_large_marker', 19: '051_large_clamp',
    20: '052_extra_large_clamp', 21: '061_foam_brick',
}

# our vocabulary word, or None to exclude this class from training entirely
YCB_TO_VOCAB = {
    '002_master_chef_can': 'can',
    '007_tuna_fish_can': None,          # too flat/small, poor shape match to our 'can' prior
    '024_bowl': 'bowl',
    '025_mug': 'mug',
    '005_tomato_soup_can': 'can',
    '006_mustard_bottle': 'bottle',
    '021_bleach_cleanser': 'bottle',
    '003_cracker_box': 'box',
    '004_sugar_box': 'box',             # was missing/mistyped -- caused sugar boxes to
                                          # be silently dropped despite belonging here,
                                          # found via a real frame where it went undetected
    '009_gelatin_box': 'box',
    '008_pudding_box': 'box',
    '061_foam_brick': 'box',
    # deliberately excluded: banana, scissors, clamps, marker, drill, wood
    # block, pitcher, power drill -- no reasonable single-superquadric
    # analog in our current vocabulary
}


def class_id_to_vocab(class_id: int):
    name = YCB_CLASS_NAMES.get(class_id)
    if name is None:
        return None
    return YCB_TO_VOCAB.get(name)