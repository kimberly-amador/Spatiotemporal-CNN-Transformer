from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Reshape, Input, Concatenate, GlobalMaxPooling1D, Dropout, Activation
from Transformer import TransformerEncoder, PositionalEmbedding


class ENCODER:
    @staticmethod
    def build(reg=l2(0.00005), shape=(384, 256), init='he_normal'):
        # Create the model
        i = Input(shape=(*shape, 1))

        # The first two layers will learn a total of 16 filters with a 3x3 kernel size
        o = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(i)
        d1 = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg, name='d1')(o)
        o = Conv2D(1, (2, 2), strides=(2, 2))(d1)  # Downsampling

        # Stack two more layers, keeping the size of each filter as 3x3 but increasing to 32 total learned filters
        o = Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)
        d2 = Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg, name='d2')(o)
        o = Conv2D(1, (2, 2), strides=(2, 2))(d2)  # Downsampling

        # Stack two more layers, keeping the size of each filter as 3x3 but increasing to 64 total learned filters
        o = Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)
        d3 = Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg, name='d3')(o)
        o = Conv2D(1, (2, 2), strides=(2, 2))(d3)  # Downsampling

        # Stack two more layers, keeping the size of each filter as 3x3 but increasing to 128 total learned filters
        o = Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)
        d4 = Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg, name='d4')(o)
        o = Conv2D(1, (2, 2), strides=(2, 2))(d4)  # Downsampling

        # Stack two more layers, keeping the size of each filter as 3x3 but increasing to 256 total learned filters
        o = Conv2D(256, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)
        d5 = Conv2D(256, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg, name='d5')(o)
        o = Conv2D(1, (2, 2), strides=(2, 2))(d5)  # Downsampling

        # Encoder outputs
        size = o.shape[1] * o.shape[2]
        output = Reshape((1, size))(o)
        down = [d1, d2, d3, d4, d5]
        return Model(inputs=i, outputs=[output, *down], name='Encoder')


class TRANSFORMER:
    @staticmethod
    def build(inputTensor, timesteps, key_dim=96, n_heads=8, n_layers=1, **kwargs):
        x = PositionalEmbedding(timesteps, inputTensor.shape[2], name='frame_position_embedding')(inputTensor)
        for _ in range(n_layers):
            x = TransformerEncoder(inputTensor.shape[2], key_dim, n_heads)(x)
        x = GlobalMaxPooling1D()(x)
        output = Dropout(0.5)(x)
        return output


class DECODER:
    @staticmethod
    def build(inputTensor, down, n_classes, reg=l2(0.00005), init='he_normal'):
        # Reshape to fit the decoder
        # TODO: automate reshape dimensions according to encoder's output
        o = Reshape((12, 8, 1))(inputTensor)

        u1 = Conv2DTranspose(1, (2, 2), strides=(2, 2))(o)  # Upsampling
        concat = Concatenate()([u1, down[4]])
        # The first layers will learn a total of 256 filters with a 3x3 kernel size
        o = Conv2D(256, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(concat)
        o = Conv2D(256, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)

        u2 = Conv2DTranspose(1, (2, 2), strides=(2, 2))(o)  # Upsampling
        concat = Concatenate()([u2, down[3]])
        # Stack two more layers, keeping the size of each filter as 3x3 but decreasing to 128 total learned filters
        o = Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(concat)
        o = Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)

        u3 = Conv2DTranspose(1, (2, 2), strides=(2, 2))(o)  # Upsampling
        concat = Concatenate()([u3, down[2]])
        # Stack two more layers, keeping the size of each filter as 3x3 but decreasing to 64 total learned filters
        o = Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(concat)
        o = Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)

        u4 = Conv2DTranspose(1, (2, 2), strides=(2, 2))(o)  # Upsampling
        concat = Concatenate()([u4, down[1]])
        # Stack two more layers, keeping the size of each filter as 3x3 but decreasing to 32 total learned filters
        o = Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(concat)
        o = Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)

        u5 = Conv2DTranspose(1, (2, 2), strides=(2, 2))(o)  # Upsampling
        concat = Concatenate()([u5, down[0]])
        # Stack three more layers, two 16 filter layers and a softmax layer
        o = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(concat)
        o = Conv2D(16, (3, 3), strides=(1, 1), padding="same", activation='relu', kernel_initializer=init, kernel_regularizer=reg)(o)
        output = Conv2D(n_classes, 1, padding="same", activation='softmax', name='Softmax')(o)
        return output
