import cv2
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util

MODEL_PATH = '/home/vangelis/faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8'
LABEL_MAP_PATH = '/home/vangelis/faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8/mscoco_label_map.pbtxt'

configs = config_util.get_configs_from_pipeline_file(MODEL_PATH + '/pipeline.config')
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)
category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH, use_display_name=True)

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(MODEL_PATH + '/checkpoint/ckpt-0').expect_partial()

# define a video capture object 
vid = cv2.VideoCapture(0) 
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

@tf.function
def detect_fn(image):
    """Detect objects in image."""
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections



while vid.isOpened():
    ret, frame = vid.read()
    if not ret:
        break

    # Convert frame to tensor
    input_tensor = tf.convert_to_tensor(
        [frame], dtype=tf.float32)

    # Run detection
    detections = detect_fn(input_tensor)

    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(int)
    detection_scores = detections['detection_scores'][0].numpy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        frame,
        detection_boxes,
        detection_classes,
        detection_scores,
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()