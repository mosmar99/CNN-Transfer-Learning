import os
import argparse
import tensorflow as tf

# tf.keras.mixed_precision.set_global_policy('mixed_float16')

image_size = (180, 180)
batch_size = 64
data_dir = "PetImages"

def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label

def make_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    previous_block_activation = x

    for kernel_count in [256, 512, 728]:
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(kernel_count, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(kernel_count, kernel_size=3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = tf.keras.layers.Conv2D(kernel_count, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = tf.keras.layers.add([x, residual])

        x = tf.keras.layers.SeparableConv2D(1024, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        if num_classes == 2:
            units = 1
        else:
            units = num_classes

        x = tf.keras.layers.Dropout(0.25)(x)

        outputs = tf.keras.layers.Dense(units, activation=None)(x)
        return tf.keras.Model(inputs, outputs)
    
def main(log_dir, epochs):

    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join("PetImages", folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = b"JFIF" in fobj.peek(10)
            finally:
                fobj.close()

            if not is_jfif:
                num_skipped += 1
                os.remove(fpath)

    print(f"Deleted {num_skipped} images.")

    ds_train = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,          
        subset="training",             
        seed=1337,                       
        image_size=image_size,         
        batch_size=batch_size,  
        label_mode="binary"            
    )

    ds_test = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,          
        subset="validation",         
        seed=1337,                      
        image_size=image_size,        
        batch_size=batch_size,         
        label_mode="binary"           
    )

    class_names = ds_train.class_names
    print(f"Class names: {class_names}")

    # ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    # ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

    ds_train = ds_train.cache()
    # ds_train = ds_train.shuffle(buffer_size=1000)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    model = make_model(input_shape=image_size + (3,), num_classes=2)

    callbacks = [
        tf.keras.callbacks.CSVLogger(os.path.join(log_dir, 'training.log')),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(log_dir, 'model_weights.h5'), save_best_only=True)
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="acc")],
    )

    model.fit(
        ds_train,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=ds_test,
    )

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                        description='Train Tensorflow model for MNIST',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('log_dir', help='Folder to write log data to')
    parser.add_argument('--epochs', help='Number of epochs to train', type=int,
                        default=10)

    args = vars(parser.parse_args())
    main(**args)


