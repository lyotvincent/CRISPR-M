
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Attention, MultiHeadAttention
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
# from multi_head_attention import MultiHeadAttention

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

def train_mnist():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype("float32") / 255


    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype("float32") / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    BATCH_SIZE = 100

    input_shape = (28, 28, 1)
    input_layer = Input(shape=input_shape)
    # main_branch = Attention()([input_layer, input_layer])
    main_branch = MultiHeadAttention(num_heads=4, key_dim=8)(input_layer, input_layer)
    main_branch = Conv2D(32, (3, 3), activation='relu')(main_branch)
    main_branch = MaxPooling2D((2, 2))(main_branch)
    # main_branch = Conv2D(64, (3, 3), activation='relu')(main_branch)
    # main_branch = MaxPooling2D((2, 2))(main_branch)
    main_branch = Conv2D(64, (3, 3), activation='relu')(main_branch)

    main_branch = Flatten()(main_branch)
    main_branch = Dense(64, activation='relu')(main_branch)
    output_layer = Dense(10, activation='softmax')(main_branch)

    model = Model(input_layer, output_layer)
    model.summary()
    model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])
    history_model = model.fit(
        train_images, train_labels,
        batch_size=BATCH_SIZE,
        epochs=5,
        validation_data=(test_images, test_labels)
    )

    print("testing...")
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("test_loss = {0}, test_acc = {1}".format(test_loss, test_acc))
    return test_acc

if __name__ == "__main__":
    test_acc = 0
    # test_acc = train_mnist()
    for i in range(50):
        test_acc += train_mnist()
    print("average test acc = %s"%(test_acc/50))

# conv + conv
#cpu# average test acc = 0.9893150001764297
#cpu# average test acc = 0.9890899986028672
#gpu# average test acc = 0.9892799985408783

# conv + conv + atten
#cpu# average test acc = 0.987419992685318
#cpu# average test acc = 0.9886350005865097
#gpu# average test acc = 0.9881860005855561

# conv + atten + conv
#cpu# average test acc = 0.9864699959754943
#cpu# average test acc = 0.9862350016832352
#gpu# average test acc = 0.9862260031700134

# att + conv + conv
#cpu# average test acc = 0.979915001988411
#gpu# average test acc = 0.9796740007400513

# conv + conv + multiheadatten
#cpu# average test acc = 0.9834399968385696
#gpu# average test acc = 0.9835759997367859

# conv + multiheadatten + conv
#cpu# average test acc = 0.9814750045537949
#gpu# average test acc = 0.9815840005874634

# multiheadatten + conv + conv
#gpu# average test acc = 0.9852100014686584

# atten + conv + multiheadatten + conv
#gpu# average test acc = 0.961002002954483
