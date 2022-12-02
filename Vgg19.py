import tensorflow as tf

from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

from tensorflow.keras import Model

from read_images import style_layers,content_layers,num_content_layers,num_style_layers

#if get_output_at(0) vgg19 with MaxPoolLayers and if get_output_at(1) vgg19 with AvgPoolLayers
def get_model():
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    vgg=replace_max_by_average_pooling(vgg)
    style_outputs = [vgg.get_layer(name).get_output_at(0) for name in style_layers]
    content_outputs = [vgg.get_layer(name).get_output_at(0) for name in content_layers]
    model_outputs = style_outputs + content_outputs
    return Model(vgg.layers[0].input, model_outputs)

def replace_max_by_average_pooling(model):
    input_layer, *other_layers = model.layers
    assert isinstance(input_layer, tf.keras.layers.InputLayer)

    x = input_layer.output
    for layer in other_layers:
        if isinstance(layer, tf.keras.layers.MaxPooling2D):
            layer = tf.keras.layers.AveragePooling2D(
                pool_size=(2,2),
                strides=(2,2),
                padding=layer.padding,
                data_format=layer.data_format,
                name=f"{layer.name}",
            )

        x = layer(x)
    return tf.keras.models.Model(inputs=input_layer.input, outputs=x)