import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from keras import layers

# Load data from Excel file
excel_path = 'C:/Users/mszab/Desktop/output_1684875991.xlsx'
df = pd.read_excel(excel_path, header=0)  # Assuming column names are in the first row

# Access the columns using the column names
X = df.iloc[:, 0].values  # Assumes the text is in the first column
y = df.iloc[:, 1].values  # Assumes the category is in the second column

# Perform label encoding on the target categories
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Reshape y to be a 1-dimensional array
y = y.reshape(-1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize the input text and convert it into sequences of integers
tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure equal length
max_sequence_length = max(len(seq) for seq in X_train)
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_sequence_length)
X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_sequence_length)

# Define the neural network architecture
model = keras.Sequential([
    layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_sequence_length),
    layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16)

# Make predictions
input_text = "Am omorat vecinul"
input_sequence = tokenizer.texts_to_sequences([input_text])
input_sequence = keras.preprocessing.sequence.pad_sequences(input_sequence, maxlen=max_sequence_length)
predicted_class = model.predict(input_sequence)[0]
predicted_category = label_encoder.inverse_transform([predicted_class])[0]
print("Predicted Category:", predicted_category)
