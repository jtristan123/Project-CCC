from tflite_model_maker import object_detector
from tflite_model_maker.object_detector import DataLoader
from tflite_model_maker.config import ExportFormat, QuantizationConfig
import tensorflow as tf
import time
import os

start_time = time.time()

# Show available devices
print("Physical devices:", tf.config.list_physical_devices())
print("GPU devices:", tf.config.list_physical_devices('GPU'))
print(dir(object_detector.DataLoader))

# GPU setup with error handling
with tf.device('/GPU:0'):

    # Load training data
    # Load Pascal VOC format data from split folders
# Create separate folders with different images
    train_data = object_detector.DataLoader.from_coco(
    'images/train',  # Put 80% of images here
    'result.json',   # Same JSON file
    label_map={0: 'cone'}
    )

    validation_data = object_detector.DataLoader.from_coco(
    'images/val',    # Put 20% of images here  
    'result.json',   # Same JSON file
    label_map={0: 'cone'}
    )

 ## Force load underlying dataset
    # Create the model
    spec = object_detector.EfficientDetLite0Spec()
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
    model.evaluate(validation_data)

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
