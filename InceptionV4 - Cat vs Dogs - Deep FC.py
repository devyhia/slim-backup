
# coding: utf-8

# In[ ]:

import CatVsDogs
import Shared

Shared.DIM = 299
CatVsDogs.DIM = 299

import tensorflow as tf
from nets import inception_v4
slim = tf.contrib.slim

parser = Shared.define_parser(klass='InceptionV4')
parser.add_argument('--depth', nargs='*', help='Special Deep Logits Architecture?')
args, unknown_args = parser.parse_known_args()


# In[10]:

class InceptionV4:
    def __init__(self, model_name, isTesting=False):
        Shared.define_model(self, model_name, self.__model)
    
    def __get_init_fn(self):
        return Shared.get_init_fn('inception_v4.ckpt', ["InceptionV4/Logits", "InceptionV4/AuxLogits"])
        
    def __model(self):
        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            self.logits, self.end_points = inception_v4.inception_v4(self.X_Norm, 2, is_training=True)
            
        if args.depth is not None and len(args.depth) > 0:
            with tf.variable_scope('InceptionV4/Logits') as scope:
                self.PreLogitsFlatten = self.end_points['PreLogitsFlatten']
                # Connect Input Layer
                self.logits = slim.fully_connected(self.PreLogitsFlatten, int(args.depth[0]))
                # Intermediary Layers
                for i in args.depth[1:]:
                    self.logits = slim.fully_connected(self.logits, int(i))
                # Connect Output Layer
                self.logits = slim.fully_connected(self.logits, 2, activation_fn=None)
    
    def train(self, sess, X, y, val_X, val_y, epochs=30, minibatch_size=50, optimizer=None):
        self.init_fn = self.__get_init_fn()
        return Shared.train_model(self, sess, X, y, val_X, val_y, epochs, minibatch_size, optimizer)
        
    def load_model(self, sess):
        return Shared.load_model(self, sess)
    
    def predict_proba(self, sess, X, step=10):
        return Shared.predict_proba(self, sess, X, step)


# In[ ]:

Shared.main(InceptionV4, args)

