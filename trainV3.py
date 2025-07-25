#IT WORKS WITH THE NEW TFLITE MODEL MAKER!!!!!!!!!!!!!!!!

#this is the lastest version of the code
#this train the model ANND exports it to exported-model-v2
#use this to train the model and export it to exported-model-v2



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
    data = DataLoader.from_pascal_voc(
        images_dir='images',
        annotations_dir='images',
        label_map={1: 'cone'}
    )

    ## Force load underlying dataset
    #data.load()

# Now access internal dataset (safe after load)
    all_data = data._data #Manually shuffle and split
     # Make it indexable
    
    random.shuffle(all_data)

    total = len(all_data)
    train_split = int(total * 0.8)
    val_split = int(total * 0.9)

    train_data = object_detector.DataLoader(all_data[:train_split])
    validation_data = object_detector.DataLoader(all_data[train_split:val_split])
    test_data = object_detector.DataLoader(all_data[val_split:])
    
    
    #train_data, test_data, validation_data = data.split(0.9)
    #train_data, validation_data = train_data.split(0.9)

    # Create the model
    spec = object_detector.EfficientDetLite0Spec()  # For Coral
    #spec = object_detector.EfficientDetLite0Spec(
    #    quantization_config=QuantizationConfig.for_int8()
    #)

    model = object_detector.create(
        train_data, 
        model_spec=spec, 
        batch_size=16, # 8 seems good too
        train_whole_model=True, 
        epochs=10, #50 is used expmle 
        validation_data=validation_data
    )

# Evaluate
    loss, accuracy = model.evaluate(train_data)

# Export
    model.export(export_dir='trainV2-exported-model-v5-81images-10epochs')

end_time = time.time()

print(f"\n⏱️ Training took {(end_time - start_time):.2f} seconds.\n")
print("✓ Training and export complete. Files saved\n")
print("2 step: Verify the model with verify_if_int8.py\n")
print("3 step: compiled with edgetpu_compiler exported-model-v6-81images-2500epochs/model.tflite\n")
print("✓ Loaded", len(train_data), "images.")