import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.models import Model
from keras.utils import plot_model
from sklearn.manifold import TSNE
from tensorflow import keras


def plt_tsne(intermediate_output, y_test, title):
    # intermediate_output = intermediate_output[:10]
    # y_test = y_test[:10]
    # train_y = pd.DataFrame(y_test)
    # dic = {0: 'NSA', 1: 'SA'}
    # ls = []
    # for index, value in train_y.iterrows():
    #     arr = np.array(value)[0]
    #     ls.append(dic[arr])
    # y_test = np.array(ls)
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(intermediate_output.reshape(len(y_test), -1))
    X_tsne_data = np.vstack((X_tsne.T, y_test)).T
    df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'label'])
    df_tsne.head()
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=df_tsne, hue='label', x='Dim1', y='Dim2', legend=False)
    # plt.title('T-SNE visualization of features')
    # plt.legend(loc='best')
    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)
    plt.savefig(title + '.png', dpi=100, bbox_inches='tight')
    plt.show()

    # embedding = umap.UMAP().fit_transform(intermediate_output.reshape(16095, -1), train_y)
    # plt.figure(figsize=(6, 6))
    # sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=ls, palette='Set1', sizes=10)
    # plt.gca().set_aspect('equal', 'datalim')
    # plt.title('Umap visualization of features')
    # plt.xlabel('umap1')
    # plt.ylabel('umap2')
    # plt.legend(loc='best')
    # plt.savefig('Umap visualization of features.pdf')
    # plt.show()


def vis_feature(log_dir, x_test, x_test_5min, y_test, recording_name_test):
    # load trained model
    weights_filepath = log_dir + '/checkpoint/classifier.weights.best.hdf5'
    custom_objects = {'tf': tf}
    my_model = keras.models.load_model(weights_filepath, custom_objects=custom_objects)
    my_model.summary()

    plot_model(my_model, to_file='./model.png', show_layer_names=True)

    # save prediction score
    intermediate_layer_model = Model(inputs=my_model.input, outputs=my_model.get_layer('input_1').output)  # 你创建新的模型
    intermediate_layer_model.summary()
    intermediate_output = intermediate_layer_model.predict([x_test, x_test_5min])  # 这个数据就是原始模型的输入数据
    plt_tsne(intermediate_output, y_test, 'input_1')

    intermediate_layer_model = Model(inputs=my_model.input,
                                     outputs=my_model.get_layer('input_2').output)  # 你创建新的模型
    intermediate_layer_model.summary()
    intermediate_output = intermediate_layer_model.predict([x_test, x_test_5min])  # 这个数据就是原始模型的输入数据
    plt_tsne(intermediate_output, y_test, 'input_2')

    intermediate_layer_model = Model(inputs=my_model.input,
                                     outputs=my_model.get_layer('activation_31').output)  # 你创建新的模型
    intermediate_layer_model.summary()
    intermediate_output = intermediate_layer_model.predict([x_test, x_test_5min])  # 这个数据就是原始模型的输入数据
    plt_tsne(intermediate_output, y_test, 'res')

    intermediate_layer_model = Model(inputs=my_model.input,
                                     outputs=my_model.get_layer('dropout_1').output)  # 你创建新的模型
    intermediate_layer_model.summary()
    intermediate_output = intermediate_layer_model.predict([x_test, x_test_5min])  # 这个数据就是原始模型的输入数据
    plt_tsne(intermediate_output, y_test, 'inormll')

    intermediate_layer_model = Model(inputs=my_model.input,
                                     outputs=my_model.get_layer('concatenate').output)  # 你创建新的模型
    intermediate_layer_model.summary()
    intermediate_output = intermediate_layer_model.predict([x_test, x_test_5min])  # 这个数据就是原始模型的输入数据
    plt_tsne(intermediate_output, y_test, 'concatenate')

    intermediate_layer_model = Model(inputs=my_model.input,
                                     outputs=my_model.get_layer('flatten').output)  # 你创建新的模型
    intermediate_layer_model.summary()
    intermediate_output = intermediate_layer_model.predict([x_test, x_test_5min])  # 这个数据就是原始模型的输入数据
    plt_tsne(intermediate_output, y_test, 'flatten')



    print()
