"""
Code adapted from:

Dalca AV, Guttag J, Sabuncu MR
Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation,
CVPR 2018

Availability: https://github.com/voxelmorph/voxelmorph/blob/legacy/ext/neuron/neuron/metrics.py
"""

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Lambda


class Dice(object):
    """
    Dice of two Tensors.

    Tensors should either be:
    1. probabilitic for each label
        i.e. [batch_size, *vol_size, nb_labels], where vol_size is the size of the volume (n-dims)
        e.g. for a 2D vol, y has 4 dimensions, where each entry is a prob for that voxel
    2. max_label
        i.e. [batch_size, *vol_size], where vol_size is the size of the volume (n-dims).
        e.g. for a 2D vol, y has 3 dimensions, where each entry is the max label of that voxel

    Variables:
        nb_labels: optional numpy array of shape (L,) where L is the number of labels. If not provided, all non-background (0) labels are computed and averaged
        weights: optional numpy array of shape (L,) giving relative weights of each label
        input_type is 'prob', or 'max_label'
        dice_type is hard or soft

    Usage:
        dice_loss = metrics.dice(weights=[1, 2, 3])
        model.compile(dice_loss, ...)
    """

    def __init__(self, nb_labels,
                 weights=None,
                 input_type='prob',
                 dice_type='soft',
                 approx_hard_max=True,
                 vox_weights=None,
                 crop_indices=None,
                 area_reg=0.1):
        """
        approx_hard_max - see note below
        [NOTE] for hard dice, we grab the most likely label and then compute a one-hot encoding for each voxel with
        respect to possible labels. To grab the most likely labels, argmax() can be used, but only when Dice is used
        as a metric. For a Dice *loss*, argmax is not differentiable, and so we can't use it. Instead, we approximate
        the prob->one_hot translation when approx_hard_max is True.
        """

        self.nb_labels = nb_labels
        self.weights = None if weights is None else K.variable(weights)
        self.vox_weights = None if vox_weights is None else K.variable(vox_weights)
        self.input_type = input_type
        self.dice_type = dice_type
        self.approx_hard_max = approx_hard_max
        self.area_reg = area_reg
        self.crop_indices = crop_indices

        if self.crop_indices is not None and vox_weights is not None:
            self.vox_weights = batch_gather(self.vox_weights, self.crop_indices)

    # Compute dice for given Tensors
    def dice(self, y_true, y_pred):

        if self.crop_indices is not None:
            y_true = batch_gather(y_true, self.crop_indices)
            y_pred = batch_gather(y_pred, self.crop_indices)

        if self.input_type == 'prob':
            # We assume that y_true is probabilistic, but just in case:
            y_true /= K.sum(y_true, axis=-1, keepdims=True)
            y_true = K.clip(y_true, K.epsilon(), 1)

            # Make sure pred is a probability
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            y_pred = K.clip(y_pred, K.epsilon(), 1)

        # Prepare the volumes to operate on
        # If we're doing 'hard' Dice, then we will prepare one-hot-based matrices of size [batch_size, nb_voxels, nb_labels],
        # where for each voxel in each batch entry, the entries are either 0 or 1.
        if self.dice_type == 'hard':

            # If given predicted probability, transform to "hard max""
            if self.input_type == 'prob':
                if self.approx_hard_max:
                    y_pred_op = _hard_max(y_pred, axis=-1)
                    y_true_op = _hard_max(y_true, axis=-1)
                else:
                    y_pred_op = _label_to_one_hot(K.argmax(y_pred, axis=-1), self.nb_labels)
                    y_true_op = _label_to_one_hot(K.argmax(y_true, axis=-1), self.nb_labels)

            # If given predicted label, transform to one-hot notation
            else:
                assert self.input_type == 'max_label'
                y_pred_op = _label_to_one_hot(y_pred, self.nb_labels)
                y_true_op = _label_to_one_hot(y_true, self.nb_labels)

        # If we're doing soft Dice, require prob output, and the data already is as we need it [batch_size, nb_voxels, nb_labels]
        else:
            assert self.input_type == 'prob', "cannot do soft dice with max_label input"
            y_pred_op = y_pred
            y_true_op = y_true

        # Return probabilities for the positive class only
        y_pred_op = Lambda(lambda_fun)(y_pred_op)[1]
        y_true_op = Lambda(lambda_fun)(y_true_op)[1]

        # Compute dice for each entry in batch. Dice will now be [batch_size, nb_labels]
        sum_dim = (1, 2)
        top = 2 * K.sum(y_true_op * y_pred_op, sum_dim)
        bottom = K.sum(K.square(y_true_op), sum_dim) + K.sum(K.square(y_pred_op), sum_dim)

        # Make sure we have no 0s on the bottom. K.epsilon()
        bottom = K.maximum(bottom, self.area_reg)
        return top / bottom

    # Weighted mean dice across all patches and labels
    def mean_dice(self, y_true, y_pred):

        # Compute dice, which will now be [batch_size, nb_labels]
        dice_metric = self.dice(y_true, y_pred)

        # Weigh the entries in the dice matrix:
        if self.weights is not None:
            dice_metric *= self.weights
        if self.vox_weights is not None:
            dice_metric *= self.vox_weights

        # Return mean_dice
        mean_dice_metric = K.mean(dice_metric)
        tf.debugging.assert_all_finite(mean_dice_metric, 'metric not finite')
        return mean_dice_metric

    # Assumes y_pred is prob (in [0,1] and sum_row = 1)
    def loss(self, y_true, y_pred):

        # Compute dice, which will now be [batch_size, nb_labels]
        dice_metric = self.dice(y_true, y_pred)

        # Loss
        dice_loss = 1 - dice_metric

        # Weigh the entries in the dice matrix:
        if self.weights is not None:
            dice_loss *= self.weights

        # Return 1 - mean_dice as loss
        mean_dice_loss = K.mean(dice_loss)
        tf.debugging.assert_all_finite(mean_dice_loss, 'Loss not finite')
        return mean_dice_loss


# -----------------------------------
#         HELPER FUNCTIONS
# -----------------------------------

def _label_to_one_hot(tens, nb_labels):
    """
    Transform a label nD Tensor to a one-hot 3D Tensor. The input tensor is first batch-flattened, and then each
    batch and each voxel gets a one-hot representation.
    """
    y = K.batch_flatten(tens)
    return K.one_hot(y, nb_labels)


def _hard_max(tens, axis):
    """
    We can't use the argmax function in a loss, as it's not differentiable. Therefore, we replace the 'hard max'
    operation (i.e. argmax + one-hot) with this approximation.
    """
    tensmax = K.max(tens, axis=axis, keepdims=True)
    eps_hot = K.maximum(tens - tensmax + K.epsilon(), 0)
    one_hot = eps_hot / K.epsilon()
    return one_hot


def batch_gather(reference, indices):
    """
    C+P From Keras pull request https://github.com/keras-team/keras/pull/6377/files

    Batchwise gathering of row indices.

    The numpy equivalent is 'reference[np.arange(batch_size), indices]', where 'batch_size' is the first dimension
    of the reference tensor.

    Arguments
        reference: A tensor with ndim >= 2 of shape (batch_size, dim1, dim2, ..., dimN)
        indices: A 1d integer tensor of shape (batch_size) satisfying 0 <= i < dim2 for each element i.

    Returns
        The selected tensor with shape (batch_size, dim2, ..., dimN).
    """
    batch_size = K.shape(reference)[0]
    indices = tf.stack([tf.range(batch_size), indices], axis=1)
    return tf.gather_nd(reference, indices)


def lambda_fun(x):
    """ Split the tensor into classes:
    For 2D images use tf.split(x, 2, 3)
    For 3D images use tf.split(x, 2, 4)
    """
    s = tf.split(x, 2, 3)
    return s
