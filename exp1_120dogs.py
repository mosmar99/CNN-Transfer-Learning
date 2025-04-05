import os
import tensorflow as tf
import matplotlib.pyplot as plt

image_size = (180, 180)
batch_size = 64
data_dir = "PetImages"

def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label

def main():
    num_skipped = 0
    for category in ("Cat", "Dog"):
        folder_path = os.path.join(data_dir, category)
        if not os.path.isdir(folder_path):
            continue

        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                with open(fpath, "rb") as fobj:
                    is_jfif = b"JFIF" in fobj.peek(10)
            except Exception:
                is_jfif = False

            if not is_jfif:
                num_skipped += 1
                os.remove(fpath)

    print(f"Deleted {num_skipped} non-JPEG images.")

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

    ds_train = ds_train.map(normalize_img).cache().prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.map(normalize_img).cache().prefetch(tf.data.AUTOTUNE)

    model = tf.keras.models.load_model('model_weights.h5')
    model.layers.pop()
    new_output = tf.keras.layers.Dense(1, activation='sigmoid')(model.layers[-1].output)
    model = tf.keras.Model(inputs=model.input, outputs=new_output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="acc")],
    )

    model.fit(
        ds_train,
        epochs=50,
        validation_data=ds_test
    )

    return ds_train, ds_test, class_names, model

if __name__ == '__main__':
    ds_train, ds_test, class_names, model = main()