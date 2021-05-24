import numpy as np
import random

import tensorflow as tf
import autokeras as ak
import kerastuner as kt

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from hasc import dataset, utils

batch = 512
epochs = 100

SEED = 0
CLASSES = 6

if __name__ == "__main__":
    print(tf.test.gpu_device_name())
    print(tf.__version__)
    
    # Set random seed
    np.random.seed(SEED)
    random.seed(SEED)
    
    # Load dataset
    x_train, y_train, x_test, y_test = dataset.load_hasc()
    
    # Reshape
    y_train = to_categorical(y_train, num_classes=CLASSES)
    y_test_ = to_categorical(y_test, num_classes=CLASSES)
    print(x_train.shape)
    print(y_test_.shape)
    
    # tf.data.Dataset
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(batch)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test_)).batch(batch)
    
    # Prepare Automodel
    # Input
    input_node = ak.Input()
    kernel_size = kt.engine.hyperparameters.Int(name="kernel_size", min_value=2, max_value=5)
    
    output_node = input_node
    
    # ConvBlock
    for i in range(0, 3):
        filter_ = kt.engine.hyperparameters.Choice(name="block_filter_{}".format(i+1), values=[16, 32, 64, 128, 256])
        layers_ = kt.engine.hyperparameters.Int(name="num_block_{}".format(i+1), min_value=1, max_value=5)
        
        output_node = ak.ConvBlock(filters=filter_, kernel_size=kernel_size, num_layers=layers_, max_pooling=True)(output_node)
    
    # Classification head
    output_node = ak.ClassificationHead(num_classes=CLASSES, multi_label=False)(output_node)
    
    # Build automodel
    auto_model = ak.AutoModel(
        inputs=input_node, outputs=output_node, overwrite=True,
        max_trials=200, objective="val_accuracy", seed=SEED,
        tuner='bayesian'
    )
    
    # Search
    auto_model.fit(train_ds, epochs=epochs, validation_data=test_ds, verbose=2)
    
    score = auto_model.evaluate(test_ds)
    print("val loss: {}".format(score[0]))
    print("val accuracy: {}".format(score[1]))
    
    # Export model
    model = auto_model.export_model()
    print(type(model.summary()))
    
    # tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True, to_file="auto_model.png")
    
    # Test
    utils.test(model)