import os
import argparse
import tensorflow as tf
from keras import layers, Model

def reset_layer_weights(layer):
    for i, weight in enumerate(layer.weights):
        initializer = tf.keras.initializers.GlorotUniform()
        layer_weights = initializer(weight.shape)
        layer.weights[i].assign(layer_weights)

def main(log_dir, epochs):
    image_size = (180, 180)
    batch_size = 64
    data_dir = "PetImages"

    ds_train = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2, subset="training", seed=1337,
        image_size=image_size, batch_size=batch_size, label_mode="binary"
    )

    ds_test = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2, subset="validation", seed=1337,
        image_size=image_size, batch_size=batch_size, label_mode="binary"
    )

    ds_train = ds_train.cache().prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.cache().prefetch(tf.data.AUTOTUNE)

    base_model = tf.keras.models.load_model('./ass1/log_1003/model_weights.h5')

    # Reset specific layer weights
    for layer_index in [27, 32]:  
        reset_layer_weights(base_model.layers[layer_index])

    x = base_model.layers[-1].input  
    new_output = layers.Dense(1, activation=None)(x)
    model = Model(base_model.input, new_output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="acc")],
    )

    callbacks = [
        tf.keras.callbacks.CSVLogger(os.path.join(log_dir, 'training.log')),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(log_dir, 'model_weights.h5'), save_best_only=True)
    ]

    model.fit(
        ds_train,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=ds_test,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train modified model on Cats vs. Dogs')
    parser.add_argument('log_dir', help='Folder to write log data to')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    args = vars(parser.parse_args())
    main(**args)
