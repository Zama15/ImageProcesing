import tensorflow as tf

print("", end='\n\n')
hello = tf.constant('Hello, TensorFlow!')
print(hello)

print(hello.numpy())


# =============================================================================

A = tf.constant(2)
B = tf.constant(3)
C = tf.constant(5)

add = tf.add(A, B)
sub = tf.subtract(A, B)
mul = tf.multiply(A, B)
div = tf.divide(A, B)

print('A:', A)
print('B:', B)
print('C:', C)

print('Addition:', add.numpy())
print('Subtraction:', sub.numpy())
print('Multiplication:', mul.numpy())
print('Division:', div.numpy())
