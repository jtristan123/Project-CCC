from tflite_model_maker import object_detector
from tflite_model_maker.object_detector import DataLoader
from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker import model_spec
#from tflite_Support import metadata
import tensorflow as tf
import time
import os
import numpy as np

start_time = time.time()

# Show available devices
print("Physical devices:", tf.config.list_physical_devices())
print("GPU devices:", tf.config.list_physical_devices('GPU'))

# Use GPU if available
with tf.device('/GPU:0'):

    # Load training data from Pascal VOC format (split folders)
    train_data = object_detector.DataLoader.from_pascal_voc(
    'custom_data/train',
    'custom_data/train/xml',
    ['cone']
    )

    val_data = object_detector.DataLoader.from_pascal_voc(
    'custom_data/validate',
    'custom_data/validate/xml',
    ['cone']
    )

# Check what was actually loaded:
    print(f"✓ Training dataset size: {len(train_data)} samples")
    print(f"✓ Validation dataset size: {len(val_data)} samples")

# Check files in directories:
    import os
    train_xmls = len([f for f in os.listdir('custom_data/train/xml') if f.endswith('.xml')])
    val_xmls = len([f for f in os.listdir('custom_data/validate/xml') if f.endswith('.xml')])
    train_imgs = len([f for f in os.listdir('custom_data/train') if f.endswith(('.jpg', '.png'))])
    val_imgs = len([f for f in os.listdir('custom_data/validate') if f.endswith(('.jpg', '.png'))])

    print(f"✓ Found {train_xmls} XML files in custom_data/train/xml")
    print(f"✓ Found {val_xmls} XML files in custom_data/validate/xml")
    print(f"✓ Found {train_imgs} images in custom_data/train")
    print(f"✓ Found {val_imgs} images in custom_data/validate")
    # Load EfficientDet Lite0 spec (Edge TPU compatible)
    #spec = object_detector.EfficientDetLite0Spec()
    spec = model_spec.get('efficientdet_lite0')
    #spec.config.use_coco_metric = True

    #spec.config.var_freeze_expr = 'efficientnet'  # optional: freeze base layers

    # Start training
    model = object_detector.create(
        train_data,
        model_spec=spec,
        batch_size=8,
        train_whole_model=True,
        epochs=10       
        #validation_data=val_data this is the error info bug

    )

    # Evaluate on test set
print("🔍 Evaluating model on test set...")
model.evaluate(val_data)

    # Export trained model
export_dir = "exported-model-coco-fixed"
model.export(export_dir=export_dir)

end_time = time.time()
print("✅ Training + Export complete.")
print(f"⏱️ Total time: {(end_time - start_time):.2f} seconds")
print(f"📁 Model exported to: {export_dir}")
