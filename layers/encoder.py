import tensorflow as tf

class Encoder(tf.Module):
  def __init__(self, in_features, output_features, name=None):
   super(Encoder, self).__init__(name=name)
   self.w = tf.Variable(
       tf.random.normal([in_features, output_features]), name='w')
   self.b = tf.Variable(tf.zeros([output_features]), name='b')
  
  def __call__(self, x):
   y = tf.matmul(x, self.w) + self.b
   return tf.nn.relu(y)



class MLP(tf.Module):
  def __init__(self, input_size, sizes, name=None):
    super(MLP, self).__init__(name=name)
    self.layers = []
    with self.name_scope:
      for size in sizes:
        self.layers.append(Encoder(in_features=input_size, output_features=size))
        input_size = size
        
  @tf.Module.with_name_scope
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
