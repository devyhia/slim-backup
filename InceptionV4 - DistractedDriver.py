
# coding: utf-8

# In[1]:

import DistractedDriver
import Shared

import tensorflow as tf
from nets import inception
slim = tf.contrib.slim

parser = Shared.define_parser(klass='InceptionV4')
parser.add_argument('--depth', nargs='*', help='Special Deep Logits Architecture?')
parser.add_argument('--which', default='original', help='Train on segmented')
args, unknown_args = parser.parse_known_args()

Shared.DIM = 299
Shared.N_CLASSES = 10

Shared.load_training_data = lambda: DistractedDriver.load_data(progressBar=True, which=args.which)


# In[ ]:

class InceptionV4:
    def __init__(self, model_name, isTesting=False):
        Shared.define_model(self, model_name, self.__model)

    def __get_init_fn(self):
        return Shared.get_init_fn('inception_v4.ckpt', ["InceptionV4/Logits", "InceptionV4/AuxLogits"])

    def __model(self):
        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(inception.inception_v4_arg_scope()):
            self.logits, self.end_points = inception.inception_v4(self.X_Norm, len(DistractedDriver.CLASSES), is_training=True)

    def train(self, sess, X, y, val_X, val_y, epochs=30, minibatch_size=50, optimizer=None):
        self.init_fn = self.__get_init_fn()
        return Shared.train_model(self, sess, X, y, val_X, val_y, epochs, minibatch_size, optimizer)

    def load_model(self, sess):
        return Shared.load_model(self, sess)

    def predict_proba(self, sess, X, step=10):
        return Shared.predict_proba(self, sess, X, step)


# In[ ]:

if __name__ == "__main__":
    Shared.main(InceptionV4, args)
