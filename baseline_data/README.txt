BASELINE COMPARISON DATA -- README

features_val_sample.npz
------------------------
Load with: data = numpy.load('features_val_sample.npz', allow_pickle=True)
  data['X']             -- (N, 13) float array, the feature vector
  data['y']              -- (N,) string array, true label
  data['feature_names']  -- (13,) string array, column names in order

Feature columns, in order:
  0: a1               -- primary radius/half-width (meters)
  1: a2                -- secondary radius/half-width (meters)
  2: eps1               -- shape exponent, vertical roundness (0=sharp corner, 1=round)
  3: eps2                -- shape exponent, horizontal roundness (0=sharp corner, 1=round)
  4: a3                   -- half-height (meters)
  5: hue                   -- color hue in degrees (0 if unavailable -- not used in this export)
  6: saturation             -- color saturation 0-1 (0 if unavailable)
  7: r_10                    -- radius at 10% of object height (meters, 0 if not axisymmetric)
  8: r_30                     -- radius at 30% of object height
  9: r_50                      -- radius at 50% of object height
  10: r_70                      -- radius at 70% of object height
  11: r_90                       -- radius at 90% of object height
  12: aspect_ratio                -- height / max(a1,a2)

NOTE: hue/saturation are all zero in this export (color intentionally
excluded to isolate pure-geometry comparison -- ping if you want a
color-included version instead). r_10..r_90 are zero for non-round
categories (box, mug, bowl, bottle) since that concept doesn't apply --
this matches exactly how our own registry treats missing features.

Labels (y): 'box', 'mug', 'bowl', 'can', 'bottle'

pointclouds_val_sample.npz (only if --include_pointclouds was used)
------------------------------------------------------------
Load with: data = numpy.load('pointclouds_val_sample.npz', allow_pickle=True)
  data['clouds']  -- object array, each element is (Mi, 3) float32 array
                     of real depth-camera points for one object instance
  data['y']        -- (N,) string array, same order as clouds, same labels

Both files use the SAME instance order and SAME split ('val_sample') as
our own reported evaluation results, so any baseline trained/tested on
this data is directly, fairly comparable to our numbers.
