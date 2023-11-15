# 1. feature_extraction_vgg16
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

# -----------------------------------------------------------------------------

# 2. fineTuning_vgg16
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Defining data preprocessing and augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize pixel values to [0, 1]
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Load and preprocess data
train_generator = datagen.flow_from_directory(
    'train_data',  # Path to training dataset
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Loading the pretrained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freezing some layers while allowing others to be trained
for layer in base_model.layers[:15]:
    layer.trainable = False
for layer in base_model.layers[15:]:
    layer.trainable = True

# Adding custom classification layers
flatten_layer = Flatten()(base_model.output)
output_layer = Dense(10, activation='softmax')(flatten_layer)

model = Model(inputs=base_model.input, outputs=output_layer)

# Compiling the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(train_generator, epochs=10)

# -----------------------------------------------------------------------------

# 3. bert

from transformers import BertTokenizer
import torch

# Loading the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Text to tokenize
text = "This is an example sentence."

# Tokenizing and pad texting
tokens = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

# Extracting input tensors
input_ids = tokens['input_ids']
attention_mask = tokens['attention_mask']

# -----------------------------------------------------------------------------

# 4. fineTuning_bert

from transformers import BertForSequenceClassification, BertTokenizer
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Loading the pretrained BERT model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Tokenization and encoding
encodings = tokenizer(your_text_data, truncation=True, padding=True)

# Building PyTorch tensors for input
input_ids = torch.tensor(encodings['input_ids'])
attention_mask = torch.tensor(encodings['attention_mask'])
labels = torch.tensor(your_labels)

# Training loop
optimizer = AdamW(model.parameters(), lr=1e-5)  # You can adjust the learning rate
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
