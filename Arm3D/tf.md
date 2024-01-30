# import
```python
import tensorflow as tf
```

# Number
## Define Constants and Variables
```python
# Define a constant
a = tf.constant(5)

# Define a variable
b = tf.Variable(10)
```

## Building a Computational Graph
```python
# Define a computational graph
c = tf.add(a, b)
c = tf.multiply(a, b)
```
# Matrix
## Creating Tensors (Matrices)
```python
# Create a 2x2 matrix using tf.constant()
matrix_a = tf.constant([[1, 2], [3, 4]])

# Create a 2x3 matrix using tf.Variable()
matrix_b = tf.Variable([[5, 6, 7], [8, 9, 10]])
```

## Matrix Multiplication
```python 
# Perform matrix multiplication
result = tf.matmul(matrix_a, matrix_b)
```

## Element-wise Operations
```python
# Element-wise addition
sum_matrix = matrix_a + matrix_b

# Element-wise subtraction
diff_matrix = matrix_a - matrix_b

# Element-wise multiplication
product_matrix = matrix_a * matrix_b

# Element-wise division
div_matrix = matrix_a / matrix_b
```

## Transpose matrix
```python 
# Transpose matrix_a
transposed_matrix = tf.transpose(matrix_a)
```

## Matrix Inverse
```python
# Compute the inverse of matrix_a
inv_matrix = tf.linalg.inv(matrix_a)
```

## Matrix Determinant
```python
# Compute the determinant of matrix_a
determinant = tf.linalg.det(matrix_a)
```

## Matrix Reductions
```python
# Compute the sum of all elements in matrix_a
sum_elements = tf.reduce_sum(matrix_a)

# Compute the mean of matrix_a along axis=0 (column-wise mean)
mean_columns = tf.reduce_mean(matrix_a, axis=0)

# Compute the maximum value along axis=1 (row-wise maximum)
max_rows = tf.reduce_max(matrix_a, axis=1)
```


# Sessions and Execution
```python
# Create a TensorFlow session
with tf.compat.v1.Session() as sess:
    # Initialize variables
    sess.run(tf.compat.v1.global_variables_initializer())

    # Execute the graph and get the result
    result = sess.run(c)
    print(result)  # Output: 15
```

#  Neural Networks 
## Define Neural Networks
```python
# Example of a simple neural network
input_layer = tf.keras.layers.Input(shape=(input_shape,))
hidden_layer = tf.keras.layers.Dense(128, activation='relu')(input_layer)
output_layer = tf.keras.layers.Dense(output_shape, activation='softmax')(hidden_layer)

model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
```

## Training the Model
```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32)
```

## Evaluation and Prediction
```python
loss, accuracy = model.evaluate(x_test, y_test)
predictions = model.predict(x_new_data)
```

# 