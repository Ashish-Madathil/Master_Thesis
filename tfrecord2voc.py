import argparse
import os
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
from glob import glob
from lxml import etree
from io import BytesIO
from collections import namedtuple
import tensorflow as tf
#tf.enable_eager_execution()


args = None

def color_map(N=256, normalized=False):
    """Generates a colormap with N unique colors

    Keyword Arguments:
        N {int} -- size of color palette (default: {256})
        normalized {bool} -- defines whether the values are between 0 - 1.0 (normalized) or 0 - 255 (default: {False})

    Returns:
        [list] -- list with N colors
    """

    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

cmap = color_map()
cmap[255] = (255, 255, 255)


def create_annotation_file(filename, image, ext, objects):
    output_path = args.dst
    basename = "%s.jpg" % filename
    root = etree.Element("annotation")
    etree.SubElement(root, "folder").text = output_path.split("/")[-1]
    etree.SubElement(root, "filename").text = basename
    s = etree.SubElement(root, "size")
    etree.SubElement(s, "width").text = str(image.shape[1])
    etree.SubElement(s, "height").text = str(image.shape[0])
    etree.SubElement(s, "depth").text = str(image.shape[2])

    for o in objects:
        o_node = etree.SubElement(root, "object")
        etree.SubElement(o_node, "name").text = o.name
        bndbox = etree.SubElement(o_node, "bndbox")
        etree.SubElement(bndbox, "xmin").text = str(o.xmin)
        etree.SubElement(bndbox, "ymin").text = str(o.ymin)
        etree.SubElement(bndbox, "xmax").text = str(o.xmax)
        etree.SubElement(bndbox, "ymax").text = str(o.ymax)

    with open("%s.xml" % os.path.join(args.annotations, filename), "w") as f:
        data = etree.tostring(root, encoding='UTF-8', xml_declaration=True, pretty_print=True)
        if isinstance(data, bytes):
            data = str(data, "utf-8")
        f.write(data)

    Image.fromarray(image).save(os.path.join(args.images,basename), quality=90)

def parse_feature(string_record):
    feature = {
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/filename': tf.FixedLenFeature([],tf.string),
        'image/source_id': tf.FixedLenFeature([],tf.string),
        'image/key/sha256': tf.FixedLenFeature([],tf.string),
        'image/encoded': tf.FixedLenFeature([],tf.string),
        'image/format': tf.FixedLenFeature([],tf.string),
        'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
        'image/object/class/text': tf.VarLenFeature(tf.string),
        'image/object/class/label': tf.VarLenFeature(tf.int64),
    }

    if args.with_masks:
        feature['image/encoded_categories'] = tf.FixedLenFeature([],tf.string)

    example = tf.parse_single_example(string_record, feature)
    return example

def parse_record():
    ds = tf.data.TFRecordDataset(args.src)
    # count = sum(1 for _ in ds)
    ds = ds.map(parse_feature, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    get_list = lambda x: tf.sparse_tensor_to_dense(x).numpy()

    obj = namedtuple("VocObject", ["name","xmin","xmax","ymin","ymax"])

    for i in tqdm(ds):
        width = i['image/width'].numpy()
        height = i['image/height'].numpy()
        filename = i['image/filename'].numpy().decode("UTF-8")
        ext = i['image/format'].numpy().decode("UTF-8")

        img = tf.image.decode_jpeg(i['image/encoded']).numpy()
        xmin = get_list(i['image/object/bbox/xmin'])
        xmax = get_list(i['image/object/bbox/xmax'])
        ymin = get_list(i['image/object/bbox/ymin'])
        ymax = get_list(i['image/object/bbox/ymax'])

        classes = i['image/object/class/text'].values.numpy()
        objects = []
        for x, y, x2, y2, label in zip(xmin, ymin, xmax, ymax, classes):
            x = int(round(x * width))
            y = int(round(y * height))
            x2 = int(round(x2 * width))
            y2 = int(round(y2 * height))

            objects += [obj(label.decode("UTF-8"),x,x2,y,y2)]

        create_annotation_file(filename, img, ext, objects)

        if args.with_masks:
            png = i['image/encoded_categories'].numpy()
            img = np.array(Image.open(BytesIO(png)))
            if type(args.mask) is np.ndarray:
                np.clip(img + args.mask, 0, 255, img)

            Image.fromarray(img).save(os.path.join(args.masks, "%s.png" % filename), "PNG")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to convert a tfrecord file used for object detection into pascal voc format')
    parser.add_argument('--src', type=str, required=True, help='Determines path to the tfrecord file')
    parser.add_argument('--dst', type=str, required=True, help='Determines output folder for converted dataset')
    parser.add_argument('--name', type=str, required=False, help='Name for database attribute in voc xml. The default value corresponds to the file name')
    parser.add_argument('--with-masks', default=False, action="store_true", help="Extract segmentation masks as well")
    parser.add_argument('--mask', type=str, required=False, help="apply a custom binary mask to segmentation masks (e.g. to make fisheye border white)")
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

    if args.with_masks:
        args.masks = os.path.join(args.dst, 'SegmentationClass')
        if not os.path.exists(args.masks):
            os.makedirs(args.masks)

    if os.path.exists(args.mask):
        args.mask = np.array(Image.open(args.mask))[:,:,0]

    parse_record()
