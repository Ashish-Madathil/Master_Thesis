# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_pascal_tf_record.py \
        --data_dir=/home/user/VOCdevkit \
        --output_path=/home/user/pascal.record
"""
from __future__ import absolute_import, division, print_function

import glob
import hashlib
import io
import logging
import os
from random import shuffle

import PIL.Image
import tensorflow as tf
from lxml import etree

from object_detection.utils import dataset_util, label_map_util
from tqdm import tqdm

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw dataset in PASCAL VOC format.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
#flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt',
#                  'Path to label map proto')
flags.DEFINE_string('label_map_path', 'pascal_label_map.pbtxt',
                   'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
flags.DEFINE_float('val_split', 0.0, 'ratio to split dataset into train/validation')
flags.DEFINE_integer('sample', 1, 'take every Nth file from file list')
flags.DEFINE_integer('frames', 0, 'number of frames to convert')
FLAGS = flags.FLAGS


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=False,
                       index=0,
                       image_subdirectory='JPEGImages',
                       mask_subdirectory='SegmentationClass'):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """

  img_path = os.path.join(image_subdirectory, data['filename'])
  full_path = os.path.join(dataset_directory, img_path)
  with tf.io.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  # feature["image/encoded_categories"] = dataset_util.bytes_feature(buffers["layer"])

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  if 'object' in data:
    for obj in data['object']:
      difficult = bool(int(obj['difficult'])) if 'difficult' in obj else False
      if ignore_difficult_instances and difficult:
        continue

      if not obj["name"] in label_map_dict:
        # print("skip label: %s" % obj["name"])
        continue

      difficult_obj.append(int(difficult))

      xmin.append(float(obj['bndbox']['xmin']) / width)
      ymin.append(float(obj['bndbox']['ymin']) / height)
      xmax.append(float(obj['bndbox']['xmax']) / width)
      ymax.append(float(obj['bndbox']['ymax']) / height)
      classes_text.append(obj['name'].encode('utf8'))
      classes.append(label_map_dict[obj['name']])
      truncated.append(int(obj['truncated']) if 'truncated' in obj else 0)
      poses.append(obj['pose'].encode('utf8') if 'pose' in obj else 'unknown'.encode('utf8'))

  feature = {
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          (str(index) + "_" + data['filename']).encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }

  mask_path = os.path.join(mask_subdirectory, data['filename'].split('.')[0] + '.png')
  full_mask_path = os.path.join(dataset_directory, mask_path)
  if os.path.exists(full_mask_path):
    with tf.io.gfile.GFile(full_mask_path, 'rb') as fid:
      encoded_png = fid.read()
      feature["image/encoded_categories"] = dataset_util.bytes_feature(encoded_png)

  example = tf.train.Example(features=tf.train.Features(feature=feature))
  return example


def create_record(record_file, annotatations, label_map_dict):
  writer = tf.io.TFRecordWriter(record_file)
  for idx, example in enumerate(tqdm(annotatations)):
    with tf.io.gfile.GFile(example, 'rb') as fid:
      xml_str = fid.read()

    try:
      xml = etree.fromstring(xml_str)
      data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
      tf_example = dict_to_tf_example(data, FLAGS.data_dir, label_map_dict,
                                      FLAGS.ignore_difficult_instances, index=idx)
      writer.write(tf_example.SerializeToString())
    except Exception as ex:
      print("exception in file: %s" %(example), ex)

  writer.close()

def main(_):
  data_dir = FLAGS.data_dir

  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
  annotations_dir = os.path.join(data_dir, FLAGS.annotations_dir)
  annotatations = sorted(glob.glob(os.path.join(annotations_dir, '*.xml')))

  if FLAGS.sample > 1:
      annotatations = annotatations[0::FLAGS.sample]

  if FLAGS.frames > 0:
      annotatations = annotatations[0:FLAGS.frames]

  shuffle(annotatations)

  datasets = {'training': {'start': 0, 'end' : len(annotatations)}}

  if FLAGS.val_split > 0.0 :
      datasets['training']['end'] = int(len(annotatations) * (1.0 - FLAGS.val_split))
      datasets['validation'] = {
        'start': datasets['training']['end'],
        'end': datasets['training']['end'] + int(len(annotatations) - datasets['training']['end'])
      }

  if not os.path.exists(FLAGS.output_path):
      os.makedirs(FLAGS.output_path)

  for k, v in datasets.items():
    print("Process %d files for %s dataset" % (v['end'] - v['start'], k))
    create_record(os.path.join(FLAGS.output_path, k +'.tfrecord'), annotatations[v['start']:v['end']], label_map_dict)

if __name__ == '__main__':
  tf.compat.v1.app.run()
