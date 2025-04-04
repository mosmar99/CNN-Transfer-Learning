import os
import argparse
import tensorflow as tf
from keras import layers
import keras

image_size = (180, 180)
batch_size = 64
data_dir = "PetImages"

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

    ds_train = ds_train.cache()
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    base_model = tf.keras.models.load_model('./ass1/log_1003/model_weights.h5')

    x = base_model.layers[-1].input  
    new_output = layers.Dense(1, activation=None)(x) 
    model = keras.Model(base_model.input, new_output)

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