import tensorflow as tf
from read_images import content_layers, style_layers, num_content_layers, num_style_layers
from Load_masks import load_seg
import os
import scipy.misc as spm
import scipy.ndimage as spi
import scipy.sparse as sps

VGG_MEAN = [103.939, 116.779, 123.68]
mean_pixel = tf.constant(VGG_MEAN)

def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)

  return result
def style_loss(style_layer, init_layer, content_seg, style_seg):
    gram_matrix_const = gram_matrix(tf.multiply(style_layer, style_seg))
    b, h, w, c = style_layer.get_shape()
    gram_matrix_var = gram_matrix(tf.multiply(init_layer, content_seg))
    return tf.reduce_mean(tf.square(gram_matrix_const - gram_matrix_var))


def content_loss(base_content, target):
    b, w, h, c = target.get_shape()
    return tf.reduce_mean(tf.square(base_content - target))


def affine_loss(output, M):
    loss_affine = 0.0
    output_t = output / 255.
    for Vc in tf.unstack(output_t, axis=-1):
        Vc_ravel = tf.reshape(tf.transpose(Vc), [-1])
        loss_affine += tf.matmul(tf.expand_dims(Vc_ravel, 0),
                                 tf.sparse.sparse_dense_matmul(M, tf.expand_dims(Vc_ravel, -1)))

    return loss_affine


def compute_loss(model, M, loss_weights, init_image, style_features, content_features, content_seg, style_seg,
                 content_width, content_height, style_width, style_height):
    style_weight, content_weight, tv_weight, affine_weight = loss_weights

    # Feed our init image through our model. This will give us the content and
    # style representations at our desired layers. Since we're using eager
    # our model is callable just like any other function!
    model_outputs = model(init_image)

    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0
    content_score = 0
    tv_score = 0
    affine_score = 0

    # Style_losses
    content_masks, style_masks = load_seg(content_seg, style_seg, [content_width, content_height],
                                          [style_width, style_height])

    _, content_seg_height, content_seg_width, _ = content_masks[0].get_shape().as_list()
    _, style_seg_height, style_seg_width, _ = style_masks[0].get_shape().as_list()
    count = 0
    weight_per_style_layer = 1.0 / float(num_style_layers)
    for style_layer, init_layer in zip(style_features, style_output_features):
        style_layer_score = 0
        for content_seg, style_seg in zip(content_masks, style_masks):
            style_layer_score += style_loss(style_layer, init_layer, content_seg, style_seg)
        style_score += weight_per_style_layer * style_layer_score
        # downsample the masks to fit to the style layers
        content_seg_width, content_seg_height = int((content_seg_width / 2)), int((content_seg_height / 2))
        style_seg_width, style_seg_height = int((style_seg_width / 2)), int((style_seg_height / 2))

        for i in range(len(content_masks)):
            content_masks[i] = tf.image.resize(content_masks[i], tf.constant((content_seg_height, content_seg_width)))
            style_masks[i] = tf.image.resize(style_masks[i], tf.constant((style_seg_height, style_seg_width)))

    # Content_losses
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * content_loss(comb_content, target_content)

    B, W, H, CH = init_image.get_shape()
    tv_score = tf.reduce_sum(tf.image.total_variation(init_image))

    style_score *= style_weight
    content_score *= content_weight
    tv_score *= tv_weight
    input_image_plus = tf.squeeze(init_image + mean_pixel, [0])
    affine_score = affine_loss(input_image_plus, M) * affine_weight
    # total loss
    loss = style_score + content_score + tv_score + affine_score
    return loss, style_score, content_score, tv_score, affine_score
