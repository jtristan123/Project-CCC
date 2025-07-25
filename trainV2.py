
from tflite_model_maker import object_detector
from tflite_model_maker.object_detector import DataLoader
from tflite_model_maker.config import QuantizationConfig
import tensorflow as tf
import time
import random


start_time = time.time()


# List all physical devices recognized by TensorFlow
print("Physical devices:", tf.config.list_physical_devices())

# Specifically list GPU devices
print("GPU devices:", tf.config.list_physical_devices('GPU'))

with tf.device('/GPU:0'):

    # Load training data
# Load Pascal VOC format data from split folders

    train_data = object_detector.DataLoader.from_pascal_voc(
    'split_data/train/images',
    'split_data/train/annotations',
    label_map={1: 'cone'}
    )

    validation_data = object_detector.DataLoader.from_pascal_voc(
    'split_data/val/images',
    'split_data/val/annotations',
    label_map={1: 'cone'}
    )

    test_data = object_detector.DataLoader.from_pascal_voc(
    'split_data/test/images',
    'split_data/test/annotations',
    label_map={1: 'cone'}
    )
 ## Force load underlying dataset


    # Create the model
    spec = object_detector.EfficientDetLite0Spec()
    spec.config.use_coco_metric = False # For Coral
    #spec = object_detector.EfficientDetLite0Spec(
    #    quantization_config=QuantizationConfig.for_int8()
    #)

    model = object_detector.create(
        train_data, 
        model_spec=spec, 
        batch_size=8, # 8 seems good too
        train_whole_model=True, 
        epochs=10, #50 is used expmle 
        validation_data=validation_data
    )

# Evaluate
    loss, accuracy = model.evaluate(test_data)

# Export
    model.export(export_dir='trainV2withsplit2.0-exported-model-v7-81images-10epochs')

end_time = time.time()

print("✓ Training + Export complete")
print(f"\n⏱️ Training took {(end_time - start_time):.2f} seconds.\n")

#print(f"✓ Final test accuracy: {accuracy:.4f}")
print("✓ Training and export complete. Files saved\n")
print("2 step: Verify the model with verify_if_int8.py\n")
print("3 step: compiled with edgetpu_compiler exported-model-v6-81images-2500epochs/model.tflite\n")
print("✓ Loaded", len(train_data), "images.")