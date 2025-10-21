# 1. Setup Paths
import os

WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'

# 2. Create Label Map
 labels = [
       {'name': 'ሀ', 'id': 1},
       {'name': 'ለ', 'id':2},
       {'name': 'መ', 'id':3},
       {'name': 'ሰ', 'id': 4},
       {'name': 'መደሰት', 'id': 5},
       {'name': 'ቋንቋ', 'id': 6},
       {'name': 'በ', 'id': 7},
       {'name': 'እናት', 'id': 8},
       {'name': 'የምልክት', 'id': 9},
       {'name': 'ይቅርታ አድርግልኝ', 'id': 10}
 ]
 with open('label_map.pbtxt', 'w', encoding='utf-8') as f:
 for label in labels:
 f.write('item { \n')
 f.write('\tname:\'{}\'\n'.format(label['name']))
 f.write('\tid:{}\n'.format(label['id']))
 f.write('}\n')
 
   
# 3. Create TF records
 !python {SCRIPTS_PATH + '/generate_tfrecord.py'} -x {IMAGE_PATH + '/train'} -l {ANNOTATION_PATH + '/label}
 !python {SCRIPTS_PATH + '/generate_tfrecord.py'} -x{IMAGE_PATH + '/test'} -l {ANNOTATION_PATH + '/label}

  
# 4. Download TF Models Pretrained Models from Tensorflow Model Zoo
 
 !cd Tensorflow && git clone https://github.com/tensorflow/models

 #wget.download('http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fp
 #!mv ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz {PRETRAINED_MODEL_PATH}
 #!cd {PRETRAINED_MODEL_PATH} && tar -zxvf ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz

# 5. Copy Model Config to Training Folder

CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
!mkdir {'Tensorflow\workspace\models\\'+CUSTOM_MODEL_NAME}
 !cp {PRETRAINED_MODEL_PATH+'/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config'} {MODEL_PAT
 
# 6. Update Config For Transfer Learning
 
 import tensorflow as tf
 from object_detection.utils import config_util
 from object_detection.protos import pipeline_pb2
 from google.protobuf import text_format
 from PIL import Image, ImageFont, ImageDraw
 
 CONFIG_PATH = MODEL_PATH+'/'+CUSTOM_MODEL_NAME+'/pipeline.config'
 
 config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
 
 config
 
 pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
 with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:                                                         
proto_str = f.read()                                                                               
text_format.Merge(proto_str, pipeline_config)

pipeline_config.model.ssd.num_classes = 10
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_MODEL_PATH+'/ssd_mobilenet_v2_fpnlite_32
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/train.re
pipeline_config.eval_input_reader[0].label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/test.r

config_text = text_format.MessageToString(pipeline_config)                                             
with tf.io.gfile.GFile(CONFIG_PATH, "wb") as f:                                                        
f.write(config_text)

# 7. Train the model
--model_dir={}/{} --pipeline_config_path={}/{}/pipeline.config --num_train_steps=20000""".format(APIMO)
 
#8 . Load Train Model From Checkpoint
 
import os
from Tensorflow.models.research.object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
 
## from object_detection.utils import visualization_utils as viz_utils
 
from object_detection.builders import model_builder
import tensorflow as tf
from object_detection.utils import config_util

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-11')).expect_partial()
@tf.function
def detect_fn(image):
  image, shapes = detection_model.preprocess(image)
  prediction_dict = detection_model.predict(image, shapes)
  detections = detection_model.postprocess(prediction_dict, shapes)
  return detections



   
