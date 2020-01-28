import numpy as np
import tensorflow as tf
import argparse
import os
from PIL import Image
from io import BytesIO
import json
from tqdm import tqdm
tf.enable_eager_execution()

args = None

def parse_feature(string_record):
    feature = {
        'image/height': tf.VarLenFeature(tf.int64),
        'image/width': tf.VarLenFeature(tf.int64),
        'image/filename': tf.FixedLenFeature([],tf.string),
        'image/format': tf.FixedLenFeature([],tf.string),
        'image/source_id': tf.FixedLenFeature([],tf.string),
        'image/encoded': tf.FixedLenFeature([],tf.string),
        #'image/encoded_categories': tf.FixedLenFeature([],tf.string)
    }
    example = tf.parse_single_example(string_record, feature)
    return example


def parse_record():
    ds = tf.data.TFRecordDataset(args.src)
    ds = ds.map(parse_feature, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    num_items = sum(1 for _ in ds)

    for i in tqdm(ds, total=num_items):
        filename = i['image/filename'].numpy().decode('utf8')
        out = os.path.join(args.dst, filename +'.png')
        Image.open(BytesIO(i['image/encoded'].numpy())).save(out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to export images located in a tfrecord file used for object')
    parser.add_argument('--src', type=str, required=True, help='Determines path to the tfrecord file')
    parser.add_argument('--dst', type=str, required=True, help='Determines output folder for exported image data')
    args = parser.parse_args()

    args.src = os.path.abspath(args.src)
    args.dst = os.path.abspath(args.dst)

    if not os.path.exists(args.src):
        raise FileNotFoundError('TFRecord file "%s" not found.' % args.src)

    if not os.path.exists(args.dst):
        os.makedirs(args.dst)

    args.annotations = os.path.join(args.dst, 'Annotations')
    if not os.path.exists(args.annotations):
        os.makedirs(args.annotations)

    args.images = os.path.join(args.dst, 'JPEGImages')
    if not os.path.exists(args.images):
        os.makedirs(args.images)


    parse_record()
