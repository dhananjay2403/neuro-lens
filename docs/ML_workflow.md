# Machine Learning Workflow (TensorFlow/Keras Based) 

## 1. Problem Definition
- Clearly define what the model is supposed to learn.
- Example: "Classify images of cats vs. dogs" or "Classify 10 categories in CIFAR-10 dataset."

---

## 2. Data Collection
- Use existing datasets or collect your own.
- Example using CIFAR-10:
  ```python
  from tensorflow.keras.datasets import cifar10
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  ```

---

## 3. Data Preprocessing

### a. Data Cleaning
- Handle missing values (if any)
- Standardize and normalize data
- Remove duplicates
- Merge and enrich data
- scikit-learn's `SimpleImputer` can be used for missing values in structured data
- Remove duplicates using pandas if applicable

### b. Data Transformation
- Normalize pixel values:
  ```python
  x_train = x_train.astype("float32") / 255.0
  x_test = x_test.astype("float32") / 255.0
  ```

- Encode categorical labels:
  ```python
  from tensorflow.keras.utils import to_categorical
  y_train = to_categorical(y_train, 10)
  y_test = to_categorical(y_test, 10)
  ```

- For tabular data: use scikit-learn transformers like `StandardScaler` or `OneHotEncoder`

### c. Data Integration
- Merging and joining data from multiple sources (if required)
- Use pandas and scikit-learn utilities to merge datasets

### d. Data Reduction (Optional)
- Dimensionality reduction (e.g., PCA)
- Sampling for large datasets
- Use scikit-learn for PCA or other dimensionality reduction:
  ```python
  from sklearn.decomposition import PCA
  pca = PCA(n_components=100)
  x_train_pca = pca.fit_transform(x_train.reshape(len(x_train), -1))
  ```

### e. Data Augmentation (to increase diversity)
  ```python
  from tensorflow.keras.preprocessing.image import ImageDataGenerator
  datagen = ImageDataGenerator(
      rotation_range=15,
      width_shift_range=0.1,
      height_shift_range=0.1,
      horizontal_flip=True
  )
  datagen.fit(x_train)
  ```

  ---


## 4. Exploratory Data Analysis (EDA)
- Analyze:
  - Data distribution
  - Missing data
  - Outliers
  - Correlations
  - Data types
- Use visualizations (e.g., matplotlib, seaborn)
- Use pandas, matplotlib, seaborn, or plotly
- Explore:
  - Class distribution
  - Correlation heatmaps (especially for tabular data)
  - Missing data using `seaborn.heatmap`
  - Use sklearn's `SelectKBest` for statistical feature analysis

---

## 5. Feature Engineering
- Combine or transform features
- Introduce domain expertise
- Reduce dimensionality (e.g., PCA)
- Rescale features
- Create new features using domain knowledge (for tabular data)
- Combine features or reduce dimensionality
- Use scikit-learn utilities like `PolynomialFeatures` or `feature_selection`

---

## 6. Data Splitting
- Train, Validation, and Test split:
  ```python
  from sklearn.model_selection import train_test_split
  x_train, x_val, y_train, y_val = train_test_split(
      x_train, y_train, test_size=0.2, random_state=42
  )
  ```

- Use scikit-learn’s `train_test_split`:
```python
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)
```

## 7. Model Building
- Choose ML type: Supervised (Classification/Regression), Unsupervised
- Build Neural Network:
  ```python
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

  model = Sequential([
      Conv2D(32, (3, 3), activation='relu', input_shape=x_train.shape[1:]),
      MaxPooling2D(2, 2),
      Conv2D(64, (3, 3), activation='relu'),
      MaxPooling2D(2, 2),
      Flatten(),
      Dense(128, activation='relu'),
      Dropout(0.3),
      Dense(10, activation='softmax')
  ])
  ```

---

## 8. Model Compilation
- Compile with optimizer, loss, and metric:
  ```python
  model.compile(
      optimizer='adam',
      loss='categorical_crossentropy',
      metrics=['accuracy']
  )
  ```

## 9. Model Training
- Train the model:
  ```python
  history = model.fit(
      datagen.flow(x_train, y_train, batch_size=64),
      validation_data=(x_val, y_val),
      epochs=20
  )
  ```

## 10. Model Evaluation
- Use both TensorFlow and scikit-learn:
```python
# TensorFlow
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# scikit-learn
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred_classes))
```

## 11. Model Optimization
### Techniques:
- Data Augmentation (already included above)
- Add Dropout or BatchNormalization layers
- Try different architectures (ResNet, MobileNet - Transfer Learning)
- Learning Rate Scheduling and Early Stopping:
  ```python
  from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

  callbacks = [
      EarlyStopping(patience=5, restore_best_weights=True),
      ReduceLROnPlateau(factor=0.5, patience=3)
  ]

  model.fit(
      datagen.flow(x_train, y_train, batch_size=64),
      validation_data=(x_val, y_val),
      epochs=20,
      callbacks=callbacks
  )
  ```

### Use Keras callbacks:
```python
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=3)
]
```

### Use scikit-learn for hyperparameter tuning with `KerasClassifier`:
```python
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

def build_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(x_train.shape[1:])),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model_cv = KerasClassifier(build_fn=build_model, epochs=10, batch_size=64)
param_grid = {'batch_size': [32, 64], 'epochs': [10, 20]}
grid = GridSearchCV(estimator=model_cv, param_grid=param_grid, cv=3)
```

---

## 12. Deployment (Optional)
```python
# Save the model
model.save("my_model.h5")

# Load it later
from tensorflow.keras.models import load_model
model = load_model("my_model.h5")
```