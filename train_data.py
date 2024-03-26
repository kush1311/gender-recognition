import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_man_dir = "/content/faces/man"
train_woman_dir = "/content/faces/woman"

datagen = ImageDataGenerator(rescale=1./255,
                             horizontal_flip=True,
                             vertical_flip=True)

train_generator = datagen.flow_from_directory(
    "/content/faces",
    target_size=(160, 160),
    batch_size=32,
    class_mode='binary'
)

base_model = tf.keras.applications.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(160, 160, 3))

for layer in base_model.layers:
    layer.trainable = False

flat = layers.Flatten()(base_model.output)

output = layers.Dense(1, activation='sigmoid')(flat)
model = models.Model(inputs=base_model.inputs, outputs=output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_generator, epochs=10)

model.save("face_data.h5")
