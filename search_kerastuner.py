import numpy as np
import random

import tensorflow as tf
import autokeras as ak
import kerastuner as kt

from datetime import datetime as dt

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Activation, GlobalAveragePooling1D, Dropout, Dense, Input
from tensorflow.keras.models import Model

from hasc import dataset, utils
from nas_activity import block

batch = 512
epochs = 100

SEED = 0
CLASSES = 6


# Build search space
def build_model(input_shape=(512, 3), num_classes=6):
    def _build_model(hp: kt.engine.hyperparameters):
        inputs = Input(shape=input_shape)
        
        # Build stem
        filters_in = 32
        output_node = Conv1D(filters_in, 3, strides=2, padding='same', use_bias=False,
                            kernel_initializer='he_normal',
                            name="stem_conv")(inputs)
        output_node = BatchNormalization(name='stem_bn')(output_node)
        output_node = Activation(activation='relu', name="stem_activation")(output_node)
        
        # Search params (overall)
        se_ratio = hp.Choice(name='se_ratio', values=[0.0, 0.25])
        
        # filters_ = [16, 32, 32, 64, 64, 128, 128]
        filters_ = [16, 32, 64, 128]
        
        for i, filters in enumerate(filters_):
            block_id = i + 1
            
            # Search params (per block)
            conv_op = hp.Choice(name="conv_ops_{}".format(block_id), 
                              values=["conv", "sep_conv", "mb_conv"])
            kernel_size = hp.Int(name="kernel_size_{}".format(block_id), 
                               min_value=2, max_value=5)

            skip_op = hp.Choice(name="skip_op_{}".format(block_id), 
                              values=["None", "pool", "identity"])
            repeats = hp.Int(name="layers_{}".format(block_id),
                           min_value=2, max_value=5)

            if skip_op == "None":
                skip_op = None 

            if conv_op == "conv":
                output_node = block.ConvBlock(
                    repeats=repeats,
                    kernel_size=kernel_size,
                    filters=filters,
                    skip_op=skip_op,
                    strides=1,
                    se_ratio=se_ratio,
                    block_id=block_id)(output_node)

                # input filters for MBConv
                filters_in = filters

            elif conv_op == "sep_conv":
                output_node = block.SeparableConvBlock(
                    repeats=repeats,
                    kernel_size=kernel_size,
                    skip_op=skip_op,
                    strides=1,
                    se_ratio=se_ratio,
                    block_id=block_id
                )(output_node)

                # input filters for MBConv
                filters_in = int(output_node.shape[-1])

            elif conv_op == "mb_conv":    
                output_node = block.MBConvBlock(
                    repeats=repeats,
                    kernel_size=kernel_size,
                    filters_in=filters_in,
                    filters_out=filters,
                    expand_ratio=1,
                    skip_op=skip_op,
                    strides=1 if i == 0 else 2,
                    se_ratio=se_ratio,
                    block_id=block_id
                )(output_node)

                # input filters for MBConv
                filters_in = filters

            # Build top
            x = Conv1D(1280,
                   1,
                   padding='same',
                   use_bias=False,
                   kernel_initializer="he_normal",
                   name="top_conv")(output_node)
            x = BatchNormalization(name="top_bn")(x)
            x = Activation(activation='relu', name="top_activation")(x)

            outputs = GlobalAveragePooling1D(name="avg_pool")(x)
            outputs = Dropout(0.2)(outputs)
            outputs = Dense(num_classes, activation="softmax", name="predictions")(outputs)

            # Create model
            model = Model(inputs=inputs, outputs=outputs)

            # Model compile
            model.compile(optimizer=Adam(learning_rate=1e-3),
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

        return model
    
    return _build_model


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
    
    # Build model
    tdatetime = dt.now()
    tuner = kt.BayesianOptimization(build_model(),
                                    objective="val_accuracy",
                                    max_trials=400,
                                    directory="auto_model",
                                    project_name="act_efficient_net_" + tdatetime.strftime('%Y%m%d%H%M%S'),
                                    seed=SEED)
    
    print(tuner.search_space_summary())
    
    # Search
    result_file = "a-nas_result_" + tdatetime.strftime('%Y%m%d%H%M%S') + ".txt"
    start = dt.now()
    with open(result_file, 'a') as f:
        print("Start: {}".format(start.strftime('%Y/%m/%d %H:%M:%S')), file=f)
    
    tuner.search(train_ds, epochs=epochs, validation_data=test_ds)
    
    end = dt.now()
    with open(result_file, 'a') as f:
        print("Finish: {}".format(end.strftime('%Y/%m/%d %H:%M:%S')), file=f)
        print("({})".format(end - start), file=f)
    
    tuner.results_summary(num_trials=1)
    
    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
    
    # Build the model with the optimal hyperparameters and train it on the data
    model = tuner.hypermodel.build(best_hps)
    model.fit(train_ds, epochs=100, validation_data=test_ds)
    
    # Best model's summary
    with open(result_file, 'a') as f:
        model.summary(print_fn=lambda x: f.write(x + "\r\n"))
        
    print(model.summary())
    
    # Test
    utils.test(model)
    