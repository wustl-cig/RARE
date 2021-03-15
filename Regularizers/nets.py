from tensorflow.python.keras.layers import Input, Concatenate, Conv2D, Conv2DTranspose, MaxPool2D, ReLU, \
    Conv3D, Conv3DTranspose, MaxPool3D, Activation, BatchNormalization, Subtract
from tensorflow.python.keras.models import Model


def dncnn(input_shape: tuple = (10, 320, 320, 2),
          depth: int = 10,
          output_channel: int = 2,
          filters=64,
          kernel_size=3):

    inpt = Input(shape=input_shape)

    x = Conv3D(filters=filters, kernel_size=kernel_size, strides=1, padding='same')(inpt)
    x = Activation('relu')(x)

    for i in range(depth - 2):
        x = Conv3D(filters=filters, kernel_size=kernel_size, strides=1, padding='same')(x)
        x = Activation('relu')(x)

    x = Conv3D(filters=output_channel, kernel_size=kernel_size, strides=1, padding='same')(x)

    x = Subtract()([inpt, x])
    model = Model(inputs=inpt, outputs=x)

    return model

def unet_3d(input_shape,
            output_channel,
            kernel_size=3,
            filters_root=32,
            conv_times=3,
            up_down_times=4):

    def conv3d_relu_dropout(input_, filters_, kernel_size_, name):
        output_ = Conv3D(filters=filters_, kernel_size=kernel_size_, padding='same', name=name+'/Conv3D')(input_)
        output_ = ReLU(name=name+'/Activation')(output_)
        return output_

    def conv3d_transpose_relu_dropout(input_, filters_, kernel_size_, name):
        output_ = Conv3DTranspose(filters=filters_, kernel_size=kernel_size_, padding='same', strides=(1, 2, 2),
                                  name=name+'/Conv3DTranspose')(input_)
        output_ = ReLU(name=name+'/Activation')(output_)
        return output_

    skip_connection = []
    ipt = Input(input_shape, name='UNet3D/Keras_Input')
    net = conv3d_relu_dropout(ipt, filters_root, kernel_size, name='UNet3D/InputConv')

    for layer in range(up_down_times):
        filters = 2 ** layer * filters_root
        for i in range(0, conv_times):
            net = conv3d_relu_dropout(net, filters, kernel_size, name='UNet3D/Down_%d/ConvLayer_%d' % (layer, i))

        skip_connection.append(net)
        net = MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='UNet3D/Down_%d/MaxPool3D' % layer)(net)

    filters = 2 ** up_down_times * filters_root
    for i in range(0, conv_times):
        net = conv3d_relu_dropout(net, filters, kernel_size, name='UNet3D/Bottom/ConvLayer_%d' % i)

    for layer in range(up_down_times - 1, -1, -1):
        filters = 2 ** layer * filters_root
        net = conv3d_transpose_relu_dropout(net, filters, kernel_size, name='UNet3D/Up_%d/UpSample' % layer)
        net = Concatenate(axis=-1, name='UNet3D/Up_%d/SkipConnection' % layer)([net, skip_connection[layer]])

        for i in range(0, conv_times):
            net = conv3d_relu_dropout(net, filters, kernel_size, name='UNet3D/Up_%d/ConvLayer_%d' % (layer, i))

    net = Conv3D(filters=output_channel, kernel_size=1, padding='same', name='UNet3D/OutputConv')(net)

    return Model(inputs=ipt, outputs=net)


def unet_2d(input_shape, kernel_size=3, filters_root=32, conv_times=3, up_down_times=5):

    def conv2d_relu_dropout(input_, filters_, kernel_size_):
        output_ = Conv2D(filters=filters_, kernel_size=kernel_size_, padding='same')(input_)
        output_ = ReLU()(output_)
        return output_

    def conv2d_transpose_relu_dropout(input_, filters_, kernel_size_):
        output_ = Conv2DTranspose(filters=filters_, kernel_size=kernel_size_, padding='same', strides=2)(input_)
        output_ = ReLU()(output_)
        return output_

    skip_layers_storage = []

    net_input = Input(input_shape)
    net = conv2d_relu_dropout(net_input, filters_root, kernel_size)

    for layer in range(up_down_times):
        filters = 2 ** layer * filters_root
        for i in range(0, conv_times):
            net = conv2d_relu_dropout(net, filters, kernel_size)

        skip_layers_storage.append(net)
        net = MaxPool2D(pool_size=2, strides=2)(net)

    filters = 2 ** up_down_times * filters_root
    for i in range(0, conv_times):
        net = conv2d_relu_dropout(net, filters, kernel_size)

    for layer in range(up_down_times - 1, -1, -1):
        filters = 2 ** layer * filters_root
        net = conv2d_transpose_relu_dropout(net, filters, kernel_size)
        net = Concatenate(axis=-1)([net, skip_layers_storage[layer]])

        for i in range(0, conv_times):
            net = conv2d_relu_dropout(net, filters, kernel_size)

    net = Conv2D(filters=1, kernel_size=1, padding='same')(net)

    return Model(inputs=net_input, outputs=net)