
# coding: utf-8

# In[1]:

import DistractedDriver
import Shared

import tensorflow as tf
from nets import alexnet
slim = tf.contrib.slim

parser = Shared.define_parser(klass='AlexNet')
parser.add_argument('--depth', nargs='*', help='Special Deep Logits Architecture?')
parser.add_argument('--which', default='original', help='Train on segmented')
args, unknown_args = parser.parse_known_args()

Shared.DIM = 299
Shared.N_CLASSES = 10
Shared.load_training_data = lambda: DistractedDriver.load_data(progressBar=True, which=args.which)

# In[ ]:

class AlexNet:
    def __init__(self, model_name, isTesting=False):
        Shared.define_model(self, model_name, self.__model)

    # def __get_init_fn(self):
    #     return Shared.get_init_fn('alexnet.ckpt', ["AlexNet/Logits", "AlexNet/AuxLogits"])

    def __model(self):
        # Resize existing images!
        self.X_Norm_Resized = tf.image.resize_images(self.X_Norm, (224,224))
        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
            self.logits, self.end_points = alexnet.alexnet_v2(self.X_Norm_Resized, len(DistractedDriver.CLASSES), is_training=True)

    def train(self, sess, X, y, val_X, val_y, epochs=30, minibatch_size=50, optimizer=None):
        # self.init_fn = self.__get_init_fn()
        self.init_fn = lambda sess: None
        return Shared.train_model(self, sess, X, y, val_X, val_y, epochs, minibatch_size, optimizer)

    def load_model(self, sess):
        return Shared.load_model(self, sess)

    def predict_proba(self, sess, X, step=10):
        return Shared.predict_proba(self, sess, X, step)


# In[ ]:

if __name__ == "__main__":
    Shared.main(AlexNet, args)
