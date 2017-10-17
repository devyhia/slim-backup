
# coding: utf-8

# In[1]:

import DistractedDriver
import Shared

import tensorflow as tf
from nets import resnet_v1
slim = tf.contrib.slim

parser = Shared.define_parser(klass='ResnetV1')
parser.add_argument('--depth', nargs='*', help='Special Deep Logits Architecture?')
parser.add_argument('--which', default='original', help='Train on segmented')
parser.add_argument('--resnet', default='50', help='Which Resnet? 50? 101? 152?')
args, unknown_args = parser.parse_known_args()

Shared.DIM = 299
Shared.N_CLASSES = 10

Shared.load_training_data = lambda: DistractedDriver.load_data(progressBar=True, which=args.which)


# In[ ]:

class ResnetV1:
    def __init__(self, model_name, isTesting=False):
        Shared.define_model(self, model_name, self.__model)

    def __get_init_fn(self):
        return Shared.get_init_fn('resnet_v1_{}.ckpt'.format(args.resnet), [
            "resnet_v1_{}/logits".format(args.resnet)
            # ,
            # "resnet_v1_{}/AuxLogits".format(args.resnet)
        ])

    def __model(self):
        # Resize existing images!
        self.X_Norm_Resized = tf.image.resize_images(self.X_Norm, (224,224))

        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            if args.resnet == '50':
                self.logits, self.end_points = resnet_v1.resnet_v1_50(self.X_Norm_Resized, len(DistractedDriver.CLASSES), is_training=True)
            elif args.resnet == '101':
                self.logits, self.end_points = resnet_v1.resnet_v1_101(self.X_Norm_Resized, len(DistractedDriver.CLASSES), is_training=True)
            elif args.resnet == '152':
                self.logits, self.end_points = resnet_v1.resnet_v1_152(self.X_Norm_Resized, len(DistractedDriver.CLASSES), is_training=True)
            else:
                raise "--resnet argument has to be either 50, 101 or 152"

    def train(self, sess, X, y, val_X, val_y, epochs=30, minibatch_size=50, optimizer=None):
        self.init_fn = self.__get_init_fn()
        return Shared.train_model(self, sess, X, y, val_X, val_y, epochs, minibatch_size, optimizer)

    def load_model(self, sess):
        return Shared.load_model(self, sess)

    def predict_proba(self, sess, X, step=10):
        return Shared.predict_proba(self, sess, X, step)


# In[ ]:

if __name__ == "__main__":
    Shared.main(ResnetV1, args)
