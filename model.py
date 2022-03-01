import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class FirstConv1x1Block(layers.Layer):
    def __init__(self, filters, groups) -> None:
        super().__init__()

        self.conv1x1 = layers.Conv2D(
            filters=filters,
            kernel_size=1,
            groups=groups,
            padding='same',
        )
        self.bn = layers.BatchNormalization()

    def call(self, x, training=False):
        x = self.conv1x1(x)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x


class SecondConv1x1Block(layers.Layer):
    def __init__(self, filters, groups):
        super().__init__()

        self.conv1x1 = layers.Conv2D(
            filters=filters,
            kernel_size=1,
            groups=groups,
            padding='same',
        )
        self.bn = layers.BatchNormalization()

    def call(self, x, training=False):
        x = self.conv1x1(x)
        x = self.bn(x, training=training)
        return x


class ShuffleUnit(layers.Layer):
    def __init__(self, filters:int, groups:int, combine_method:str, grouped=True):
        super(ShuffleUnit, self).__init__()

        if not isinstance(filters, int):
            raise TypeError("''filters'' must be an int.")
        if groups not in [1, 2, 3, 4, 8]:
            raise ValueError("''groups'' need to be one of the following numbers: [1, 2, 3, 4, 8]")
        if combine_method not in ['concat', 'add']:
            raise ValueError("''combine_method'' must be ''concat'' or ''add''.")
        if not isinstance(grouped, bool):
            raise TypeError("''grouped'' must be a bool.")
        
        self.filters = filters
        self.combine = combine_method
        self.grouped_conv = grouped
        self.groups = groups if self.grouped_conv else 1

        bottleneck_filters = self.filters // 4
        
        self.first_conv_1x1 = FirstConv1x1Block(
            filters=bottleneck_filters,
            groups=self.groups,
        )
        if self.combine == 'concat':
            dw_stride = 2
            self.avg_pool = layers.AveragePooling2D(
                pool_size=3, 
                strides=dw_stride, 
                padding='same'
            )
        else:
            dw_stride = 1
        self.DWConv_3x3 = layers.DepthwiseConv2D(
            kernel_size=3, 
            strides=dw_stride, 
            padding='same'
        )
        self.bn = layers.BatchNormalization()
        self.second_conv_1x1 = SecondConv1x1Block(
            filters=self.filters, 
            groups=self.groups,
        )


    def channel_shuffle(self, inputs):
        n, h, w, c = inputs.shape.as_list()
        x_reshaped = tf.reshape(inputs, [-1, h, w, self.groups, c // self.groups])
        x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
        output = tf.reshape(x_transposed, [-1, h, w, c])
        return output

    def combine_func(self, x, y):
        if self.combine == 'add':
            return tf.add(x, y)
        else: # concat
            return tf.concat([x, y], axis=-1)

    def call(self, inputs, training=False):
        residuals = inputs
        if self.combine == 'concat':
            residuals = self.avg_pool(residuals)

        x = self.first_conv_1x1(inputs, training=training)
        if self.grouped_conv:
            x = self.channel_shuffle(x)
        x = self.DWConv_3x3(x)
        x = self.bn(x, training=training)
        if not self.grouped_conv:
            x = tf.nn.relu(x)
        x = self.second_conv_1x1(x, training=training)

        self.combine_func(residuals, x)
        return tf.nn.relu(x)


class ShuffleNet(keras.Model):
    def __init__(self, groups, num_classes=1000) -> None:
        super(ShuffleNet, self).__init__()

        if groups not in [1, 2, 3, 4, 8]:
            raise ValueError("Groups need to be one of the following numbers: [1, 2, 3, 4, 8]")
        self.groups = groups

        if groups == 1:
            self.stage_filters = [-1, 24, 144, 288, 567]
        elif groups == 2:
            self.stage_filters = [-1, 24, 200, 400, 800]
        elif groups == 3:
            self.stage_filters = [-1, 24, 240, 480, 960]
        elif groups == 4:
            self.stage_filters = [-1, 24, 272, 544, 1088]
        else:
            self.stage_filters = [-1, 24, 384, 768, 1536]
        self.repeats = [3, 7, 3]

        self.conv1          = layers.Conv2D(filters=24, kernel_size=3, strides=2, padding='same')
        self.max_pool       = layers.MaxPool2D(pool_size=3, strides=2, padding='same')
        self.stage2         = self.make_stage(stage=2)
        self.stage3         = self.make_stage(stage=3)
        self.stage4         = self.make_stage(stage=4)
        self.global_pool    = layers.GlobalAveragePooling2D()
        self.fc             = layers.Dense(num_classes)


    def make_stage(self, stage):
        units = []
        stage_filters = self.stage_filters[stage]
        units.append(
            ShuffleUnit(
                filters=stage_filters,
                groups=self.groups,
                combine_method='concat',
                grouped=True,
            )
        )

        for _ in range(self.repeats[stage-2]):
            units.append(
                ShuffleUnit(
                    filters=stage_filters,
                    groups=self.groups,
                    combine_method='add',
                    grouped=True,
                )
            )   
        name = f"stage_{stage}"
        stage = keras.Sequential(units, name=name)
        return stage

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.max_pool(x)
        x = self.stage2(x, training=training)
        x = self.stage3(x, training=training)
        x = self.stage4(x, training=training)
        x = self.global_pool(x)
        x = self.fc(x)
        return x

    def model(self, input_shape):
        x = keras.Input(shape=input_shape, dtype=tf.float32)
        return keras.Model(inputs=[x], outputs=self.call(x))