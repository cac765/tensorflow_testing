"""
  IMPORTS
===========================================================================
"""
import matplotlib
import matplotlib.pyplot as plt

import os
import pathlib
import logging
import random
import io
import imageio
import glob
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, Javascript
from IPython.display import Image as IPyImage

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
#from object_detection.utils import colab_utils
from object_detection.utils import config_util
from object_detection.builders import model_builder


"""
  UTILITIES
===========================================================================
"""
logging.basicConfig( level=logging.INFO )
logging.info( "Logger configured for user output." )
logging.info( "Imported modules successfully, defining utilities..." )

def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path.

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    image_name=None):
  """Wrapper function to visualize detections.

  Args:
    image_np: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    figsize: size for the figure.
    image_name: a name for the image file.
  """
  image_np_with_annotations = image_np.copy()
  viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_annotations,
      boxes,
      classes,
      scores,
      category_index,
      use_normalized_coordinates=True,
      min_score_thresh=0.8)
  if image_name:
    plt.imsave(image_name, image_np_with_annotations)
  else:
    plt.imshow(image_np_with_annotations)


"""
  DATA VISUALIZATION
===========================================================================
"""
logging.info( "Loading images as numpy arrays..." )

# Load images and visualize
while "models" in pathlib.Path.cwd().parts:
  os.chdir("..")
path = os.path.join(os.getcwd(), "models/research")
train_image_dir = "object_detection/test_images/person/test/"
train_images_np = []
for index in range(1, 5):
  image_path = os.path.join(path, train_image_dir + "t" + str(index) + ".jpg")
  train_images_np.append(load_image_into_numpy_array(image_path))

plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.labelsize'] = False
plt.rcParams['ytick.labelsize'] = False
plt.rcParams['xtick.top'] = False
plt.rcParams['xtick.bottom'] = False
plt.rcParams['ytick.left'] = False
plt.rcParams['ytick.right'] = False
plt.rcParams['figure.figsize'] = [14, 7]

logging.info( "Plotting images..." )

for index, train_image_np in enumerate(train_images_np):
  plt.subplot(2, 2, index + 1)
  plt.imshow(train_image_np)
plt.show()


"""
  LOAD PRE-TRAINED MODEL
===========================================================================
"""
logging.info( "Loading pre-trained model..." )

full_tf_model_dir = os.path.join( 
    path, "object_detection/ssd_mobilenet_v2/saved_model")
detection_model = tf.saved_model.load( full_tf_model_dir )

"""
  MODEL CONVERSION
===========================================================================
"""
# Model has already been converted to a .tflite format via CLI!!!


"""
  TEST .tflite MODEL
===========================================================================
"""
logging.info( "Building full detection model for preprocessing..." )

pipeline_config = os.path.join(
    path, "object_detection/ssd_mobilenet_v2/pipeline.config" )
configs = config_util.get_configs_from_pipeline_file( pipeline_config )
model_config = configs["model"]
#model_config.ssd.num_classes = 90 ## This line commented because we are not at this time making changes to the model.
model_config.ssd.freeze_batchnorm = True
detection_model = model_builder.build(
    model_config=model_config, is_training=False)

logging.info( "Converting test images to numpy arrays..." )

test_image_dir = os.path.join(
    path, "object_detection/test_images/person/test/" )
test_images_np = []
for index in range(1, len( os.listdir( test_image_dir )) + 1 ):
  image_path = os.path.join(test_image_dir, 't' + str(index) + '.jpg')
  test_images_np.append(np.expand_dims(
      load_image_into_numpy_array(image_path), axis=0))

# Again, uncomment this decorator if you want to run inference eagerly
def detect(interpreter, input_tensor):
  """Run detection on an input image.

  Args:
    interpreter: tf.lite.Interpreter
    input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
      Note that height and width can be anything since the image will be
      immediately resized according to the needs of the model within this
      function.

  Returns:
    A dict containing 3 Tensors (`detection_boxes`, `detection_classes`,
      and `detection_scores`).
  """
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # We use the original model for pre-processing, since the TFLite model doesn't
  # include pre-processing.
  preprocessed_image, shapes = detection_model.preprocess(input_tensor)
  interpreter.set_tensor(input_details[0]['index'], preprocessed_image.numpy())

  interpreter.invoke()

  boxes = interpreter.get_tensor(output_details[0]['index'])
  classes = interpreter.get_tensor(output_details[1]['index'])
  scores = interpreter.get_tensor(output_details[2]['index'])
  return boxes, classes, scores

logging.info( "Loading TFLite model..." )

# Load the TFLite model and allocate tensors.
model_path = os.path.join(
    path, "object_detection/tflite/model.tflite" )
interpreter = tf.lite.Interpreter(
    model_path=model_path)
interpreter.allocate_tensors()

person_class_id = 1
category_index = { person_class_id: {"id": person_class_id, "name": "person" }}

# Note that the first frame will trigger tracing of the tf.function, which will
# take some time, after which inference should be fast.
logging.info( "Making predictions and plotting..." )
label_id_offset = 1
for i in range(len(test_images_np)):
  input_tensor = tf.convert_to_tensor(test_images_np[i], dtype=tf.float32)
  boxes, classes, scores = detect(interpreter, input_tensor)

  output_path = os.path.join(
    path, "object_detection/test_images/person/output" )
  
  plot_detections(
      test_images_np[i][0],
      boxes[0],
      classes[0].astype(np.uint32) + label_id_offset,
      scores[0],
      category_index, 
      figsize=(15, 20),
      image_name=output_path + "/out" + str(i+1) + ".jpg" )
  plt.show()

logging.info( "PROGRAM END." )


    
    
