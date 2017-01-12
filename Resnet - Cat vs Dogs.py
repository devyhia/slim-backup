
# coding: utf-8

# In[4]:

import CatVsDogs
import Shared

Shared.DIM = 299
CatVsDogs.DIM = 299

import tensorflow as tf
from nets import resnet_v1
slim = tf.contrib.slim

parser = Shared.define_parser(klass='Resnet')
parser.add_argument('--version', default='101', help='Resnet Version: 101 or 152')
args, unknown_args = parser.parse_known_args()


# In[11]:

class Resnet:
    def __init__(self, model_name, isTesting=False):
        Shared.define_model(self, model_name, self.__model)
    
    def __get_init_fn(self):
        return Shared.get_init_fn('resnet_v1_{}.ckpt'.format(args.version), ["resnet_v1_{}/logits".format(args.version)])
        
    def __model(self):
        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            if args.version == '101':
                self.logits, self.end_points = resnet_v1.resnet_v1_101(self.X_Norm, 2, is_training=True)
            else:
                self.logits, self.end_points = resnet_v1.resnet_v1_152(self.X_Norm, 2, is_training=True)
    
    def train(self, sess, X, y, val_X, val_y, epochs=30, minibatch_size=50, optimizer=None):
        self.init_fn = self.__get_init_fn()
        return Shared.train_model(self, sess, X, y, val_X, val_y, epochs, minibatch_size, optimizer)
        
    def load_model(self, sess):
        return Shared.load_model(self, sess)
    
    def predict_proba(self, sess, X, step=10):
        return Shared.predict_proba(self, sess, X, step)


# In[ ]:

Shared.main(Resnet, args)

