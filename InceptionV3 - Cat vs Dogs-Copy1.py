
# coding: utf-8

# In[1]:

import CatVsDogs
import Shared

Shared.DIM = 299
CatVsDogs.DIM = 299

import tensorflow as tf
from nets import inception_v3
slim = tf.contrib.slim

parser = Shared.define_parser(klass='InceptionV3')
parser.add_argument('--deep-logits', dest='deep_logits', default=False, action='store_true', help='Deep Logits?')
parser.add_argument('--depth', nargs='*', help='Special Deep Logits Architecture?')
parser.add_argument('--aux-logits', dest='aux_logits', default=False, action='store_true', help='Deep Aux Logits?')
args, unknown_args = parser.parse_known_args()


# In[ ]:

class InceptionV3:
    def __init__(self, model_name, isTesting=False):
        Shared.define_model(self, model_name, self.__model)
    
    def __get_init_fn(self):
        return Shared.get_init_fn('inception_v3.ckpt', ["InceptionV3/Logits", "InceptionV3/AuxLogits"])
        
    def __model(self):
        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            self.logits, self.end_points = inception_v3.inception_v3(self.X_Norm, 2, is_training=True)
        
        if args.deep_logits:
            with tf.variable_scope('InceptionV3/Logits') as scope:
                self.PreLogitsFlatten = slim.flatten(self.end_points['PreLogits'])
                self.logits = slim.fully_connected(self.PreLogitsFlatten, 200)
                self.logits = slim.fully_connected(self.logits, 2, activation_fn=None)
        
        if args.depth is not None and len(args.depth) > 0:
            with tf.variable_scope('InceptionV3/Logits') as scope:
                self.PreLogitsFlatten = slim.flatten(self.end_points['PreLogits'])
                # Connect Input Layer
                self.logits = slim.fully_connected(self.PreLogitsFlatten, int(args.depth[0])) # scope='fc0'
                # Intermediary Layers
                for idx, i in enumerate(args.depth[1:]):
                    self.logits = slim.fully_connected(self.logits, int(i)) # scope='fc{}'.format(idx)
                # Connect Output Layer
                self.logits = slim.fully_connected(self.logits, 2, activation_fn=None) # scope='fc{}'.format(len(args.depth))
        
        if args.aux_logits:
            self.logits = self.logits + self.end_points['AuxLogits']
    
    def train(self, sess, X, y, val_X, val_y, epochs=30, minibatch_size=50, optimizer=None):
        self.init_fn = self.__get_init_fn()
        return Shared.train_model(self, sess, X, y, val_X, val_y, epochs, minibatch_size, optimizer)
        
    def load_model(self, sess):
        return Shared.load_model(self, sess)
    
    def predict_proba(self, sess, X, step=10):
        return Shared.predict_proba(self, sess, X, step)


# In[ ]:

if __name__ == "__main__":
    Shared.main(InceptionV3, args)

