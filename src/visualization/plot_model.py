from keras.utils import plot_model

from src.infra.model import create_sp_classifier

classifier = create_sp_classifier()

plot_model(classifier, to_file='./model.png', show_shapes=True, show_layer_names=True, show_layer_activations=True,
           rankdir='LR', expand_nested=True)
