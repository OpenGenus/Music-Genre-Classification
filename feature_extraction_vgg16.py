from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Defining data preprocessing and augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalizing pixel values to [0, 1]
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Loading and preprocessing data
train_generator = datagen.flow_from_directory(
    'train_data',  # Path to training dataset
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Loading pretrained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freezing the layers of the pretrained model
for layer in base_model.layers:
    layer.trainable = False

# Adding custom classification layers
flatten_layer = Flatten()(base_model.output)
output_layer = Dense(10, activation='softmax')(flatten_layer)

model = Model(inputs=base_model.input, outputs=output_layer)

# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(train_generator, epochs=10)
