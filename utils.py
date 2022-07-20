import numpy as np

def Parameter_GetCount():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)

def generate_numvec(digit, z = None):
    out = np.zeros((1, 12))
    out[:, digit + 2] = 1.
    if z is None:
        return out
    else:
        for i in range(len(z)):
            out[:,i] = z[i]
        return out

def merge3(images,size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h, w * size,3))

    for idx, image in enumerate(images):
        j = int(idx)
        img[0:h, int(j*h):int(j*h+h),0:3] = image

    return img

def bernoullisample(x):
    return np.random.binomial(1, x, size=x.shape).astype('float32')


def merge2(images,size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1],3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        j = int(j)

        img[int(j*h):int(j*h+h), int(i*w):int(i*w+w),0:3] = image

    return img

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        j = int(j)

        img[int(j*h):int(j*h+h), int(i*w):int(i*w+w)] = image

    return img

from random import shuffle
import scipy.misc
import numpy as np

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    # crop the images to [crop_h,crop_w,3] then resize to [resize_h,resize_w,3]
    if crop_w is None:
        crop_w = crop_h # the width and height after cropped
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def merge55(images, size):
    # merge all output images(of sample size:8*8 output images of size 64*64) into one big image
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images): # idx=0,1,2,...,63
        i = idx % size[1] # column number
        j = idx // size[1] # row number
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

def transform(image, npx=64, is_crop=True, resize_w=64):
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.  # change pixel value range from [0,255] to [-1,1] to feed into CNN

def inverse_transform(images):
    return (images+1.)/2. # change image pixel value(outputs from tanh in range [-1,1]) back to [0,1]

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float) # [width,height] flatten RGB image to grayscale image
    else:
        return scipy.misc.imread(path).astype(np.float) # [width,height,color_dim]

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def get_image2(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    # size indicates how to arrange the images to form a big summary image
    # images: [batchsize,height,width,color]
    # example: save_images(img, [8, 8],'./{}/train_{:02d}_{:04d}.png'.format(FLAGS.sample_dir, epoch, idx))
    return imsave(inverse_transform(images), size, image_path)

def save_images_256(images, size, image_path):
    images = inverse_transform(images)
    h, w = 64, 64 # 256,256
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images): # idx=0,1,2,...,63
        image = scipy.misc.imresize(image,[h,w])
        i = idx % size[1] # column number
        j = idx // size[1] # row number
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return scipy.misc.imsave(image_path, img)

from absl import logging
import tensorflow.compat.v1 as tf
#import tensorflow_probability as tfp

def generate_gaussian(logits, sigma_nonlin, sigma_param):
  """Generate a Gaussian distribution given a selected parameterisation."""

  mu, sigma = tf.split(value=logits, num_or_size_splits=2, axis=1)

  if sigma_nonlin == 'exp':
    sigma = tf.exp(sigma)
  elif sigma_nonlin == 'softplus':
    sigma = tf.nn.softplus(sigma)
  else:
    raise ValueError('Unknown sigma_nonlin {}'.format(sigma_nonlin))

  if sigma_param == 'var':
    sigma = tf.sqrt(sigma)
  elif sigma_param != 'std':
    raise ValueError('Unknown sigma_param {}'.format(sigma_param))

  return tfp.distributions.Normal(loc=mu, scale=sigma)


def construct_prior_probs(batch_size, n_y, n_y_active):
  """Construct the uniform prior probabilities.

  Args:
    batch_size: int, the size of the batch.
    n_y: int, the number of categorical cluster components.
    n_y_active: tf.Variable, the number of components that are currently in use.

  Returns:
    Tensor representing the prior probability matrix, size of [batch_size, n_y].
  """
  probs = tf.ones((batch_size, n_y_active)) / tf.cast(
      n_y_active, dtype=tf.float32)
  paddings1 = tf.stack([tf.constant(0), tf.constant(0)], axis=0)
  paddings2 = tf.stack([tf.constant(0), n_y - n_y_active], axis=0)
  paddings = tf.stack([paddings1, paddings2], axis=1)
  probs = tf.pad(probs, paddings, constant_values=1e-12)
  probs.set_shape((batch_size, n_y))
  logging.info('Prior shape: %s', str(probs.shape))
  return probs


def maybe_center_crop(layer, target_hw):
  """Center crop the layer to match a target shape."""
  l_height, l_width = layer.shape.as_list()[1:3]
  t_height, t_width = target_hw
  assert t_height <= l_height and t_width <= l_width

  if (l_height - t_height) % 2 != 0 or (l_width - t_width) % 2 != 0:
    logging.warn(
        'It is impossible to center-crop [%d, %d] into [%d, %d].'
        ' Crop will be uneven.', t_height, t_width, l_height, l_width)

  border = int((l_height - t_height) / 2)
  x_0, x_1 = border, l_height - border
  border = int((l_width - t_width) / 2)
  y_0, y_1 = border, l_width - border
  layer_cropped = layer[:, x_0:x_1, y_0:y_1, :]
  return layer_cropped