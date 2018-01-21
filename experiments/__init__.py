from __future__ import division

from keras.applications.resnet50 import ResNet50
from keras.regularizers import l2

from keras.applications import vgg16
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers import Dense, Flatten
from keras.constraints import maxnorm
from keras.models import Model
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import GlobalAveragePooling2D


def resNet50(weights='imagenet', img_width=224, img_height=224):
    if weights == 'imagenet':
        base_model = ResNet50(weights=weights, include_top=False, input_shape=(img_width, img_height, 3))
    else:
        base_model = ResNet50(weights=None, include_top=False, input_shape=(img_width, img_height, 3))

    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(21, activation='softmax', name='fc21')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    if weights and weights != 'imagenet':
        model.load_weights(weights)
    return model


def resNet50_second_phase(weights='imagenet', img_width=224, img_height=224):
    if weights == 'imagenet':
        base_model = ResNet50(weights=weights, include_top=False, input_shape=(img_width, img_height, 3))
    else:
        base_model = ResNet50(weights=None, include_top=False, input_shape=(img_width, img_height, 3))

    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(21, activation='softmax', name='fc21')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in model.layers[:163]:
        layer.trainable = False
    for layer in model.layers[163:]:
        layer.trainable = True

    if weights and weights != 'imagenet':
        model.load_weights(weights)
    return model


def inceptionV3_first_phase_model(weights='imagenet', img_width=299, img_height=299):
    if weights == 'imagenet':
        base_model = InceptionV3(weights=weights, include_top=False, input_shape=(img_width, img_height, 3))
    else:
        base_model = InceptionV3(weights=None, include_top=False, input_shape=(img_width, img_height, 3))

    x = base_model.output
    x = Dropout(0.4)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    predictions = Dense(21, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    if weights and weights != 'imagenet':
        model.load_weights(weights)
    return model


def inceptionV3_second_phase_model(weights=None, img_width=299, img_height=299):
    if weights == 'imagenet':
        base_model = InceptionV3(weights=weights, include_top=False, input_shape=(img_width, img_height, 3))
    else:
        base_model = InceptionV3(weights=None, include_top=False, input_shape=(img_width, img_height, 3))

    x = base_model.output
    x = Dropout(0.4)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    predictions = Dense(21, activation='softmax', W_regularizer=l2(.01))(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in model.layers[:279]:
        layer.trainable = False
    for layer in model.layers[279:]:
        layer.trainable = True

    if weights and weights != 'imagenet':
        model.load_weights(weights)
    return model


def vgg16_first_phase_model(weights='imagenet', img_width=224, img_height=224):
    base_model = vgg16.VGG16(include_top=False, weights=weights, input_shape=(img_width, img_height, 3))

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1', kernel_constraint=maxnorm(2.))(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2', kernel_constraint=maxnorm(2.))(x)
    x = Dropout(0.5)(x)
    x = Dense(21, activation='softmax', name='predictions')(x)

    return Model(name='VGG-16', inputs=[base_model.input], outputs=[x])


def vgg16_second_phase_model(weights=None, img_width=224, img_height=224):
    model = vgg16_first_phase_model(None, img_width, img_height)

    for layer in model.layers[15:19]:
        layer.trainable = True

    if weights:
        model.load_weights(weights)
    return model


def resNet50features_plus_lstm(feature_vector_length=2048, weights=None, timestep=10):

    model = Sequential(name='resNet50_features+lstm');
    # Classification block
    model.add(TimeDistributed(Dropout(0.5), input_shape=(timestep, feature_vector_length)))
    model.add(LSTM(128, name='lstm1', return_sequences=True))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(Dense(21, activation='softmax'), name='predictions'))

    if weights:
        model.load_weights(weights)
    return model


def inceptionV3features_plus_lstm(feature_vector_length=2048, weights=None, timestep=10):

    model = Sequential(name='inceptionV3_features+lstm');
    # Classification block
    model.add(TimeDistributed(Dense(512, activation='relu', name='fc1'), input_shape=(timestep, feature_vector_length)))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(LSTM(256, name='lstm1', return_sequences=True))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(Dense(21, activation='softmax'), name='predictions'))

    if weights:
        model.load_weights(weights)
    return model


def probabilities_plus_lstm(feature_vector_length=21, weights=None, batch_size=10, timestep=10):

    model = Sequential(name='probabilities_features+lstm');
    # Classification block
    model.add(TimeDistributed(Dropout(0.5), input_shape=(timestep, feature_vector_length)))
    model.add(LSTM(32, name='lstm1', return_sequences=True))
    model.add(TimeDistributed(Dense(21, activation='softmax'), name='predictions'))

    if weights:
        model.load_weights(weights)
    return model
