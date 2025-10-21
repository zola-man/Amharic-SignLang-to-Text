# 1. Detect from an Image
import cv2 
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
 
category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')
 
IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'test', 'ሰ.39c19675-0aac-11ee-954f-9061aec70622.jpg')

from PIL import Image
 import numpy as np
 import tensorflow as tf
 import matplotlib.pyplot as plt
 import cv2
 from object_detection.utils import label_map_util
 from Tensorflow.models.research.object_detection.utils import visualization_utils as viz_utils
 # Path to label map file
 PATH_TO_LABELS = 'D:\RealTimeObjectDetection\label_map.pbtxt'
 # Load the label map
 category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=Tr
 # Path to the image file
 IMAGE_PATH = 'D:/RealTimeObjectDetection/Tensorflow/workspace/images/test/መ.02b9cf82-0ab5-11ee-a770-90
 # Load the image using PIL with the correct color mode
 img = Image.open(IMAGE_PATH).convert('RGB')
 # Convert the image to a NumPy array
 image_np = np.array(img)
 # Convert the NumPy array back to a PIL image
 img = Image.fromarray(image_np)
 # Convert the PIL image to a TensorFlow tensor and apply object detection
 input_tensor = tf.convert_to_tensor(np.expand_dims(img, 0), dtype=tf.float32)
 detections = detect_fn(input_tensor)
 # Process the detection results
 num_detections = int(detections.pop('num_detections'))
 detections = {key: value[0, :num_detections].numpy()
 for key, value in detections.items()}
 detections['num_detections'] = num_detections
 # detection_classes should be ints.
 detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
 label_id_offset = 1
 image_np_with_detections = image_np.copy()
 # Visualize the detection results on the original image
 viz_utils.visualize_boxes_and_labels_on_image_array(
 image_np_with_detections,
 detections['detection_boxes'],
 detections['detection_classes'] + label_id_offset,
 detections['detection_scores'],
 category_index,
 use_normalized_coordinates=True,
 max_boxes_to_draw=5,
 min_score_thresh=.8,
 agnostic_mode=False)
 # Display the image using matplotlib.pyplot.imshow()
 plt.imshow(image_np_with_detections)
 plt.show()

# 2. Detect in Real-Time

import cv2 
import numpy as np
 category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')
 # Setup capture
 cap = cv2.VideoCapture(0)
 width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
 height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
