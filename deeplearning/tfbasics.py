import tensorflow as tf

x1 = tf.constant(5)
x2 = tf.constant(6)

#This does work
#result = x1 * x2

#This is more efficient
result = tf.multiply(x1,x2)
print(result)

'''
sess = tf.Session()
print(sess.run(result))
sess.close()
'''

#Another way to do it, will automatically close sess
with tf.Session() as sess:
	print(sess.run(result))