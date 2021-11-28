##############################################3
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2


#훈련후, 그 모델 유지하면서 model.predict만 하는법 - model 학습 15_01_참고해서 모델빌드

def PreTrained_save_model():
    import keras.preprocessing.image
    from tensorflow.keras import layers
    import tensorflow.keras as keras

    gen_train = keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255,  # scailing 은 각각의 피쳐가 균등하게 만든다
                                                             zoom_range=1.5)  # ctrl써서 들어가보면 그 참고가 있다.
    # 그거 읽어보면서 데이터 증강을 할 수 있다.
    # preprocessing function등등...

    flow_train = gen_train.flow_from_directory('plastic_recy3/train',  # ctrl써서 들어가본다. 그리고 안에 필요한거를 채운다.
                                               target_size=[224, 224],
                                               class_mode='sparse')  # categorical, binary, sparse' 기본 코드 xx

    get_test = keras.preprocessing.image.ImageDataGenerator()
    flow_test = get_test.flow_from_directory('plastic_recy3/test',
                                             target_size=[224, 224],
                                             class_mode='sparse')
    # exit() #여기서 쓰면 잘 불러왔는지 아닌지 알 수 있다!!

    conv_base = keras.applications.VGG16(include_top=False,
                                         input_shape=[224, 224, 3])  # top은 VGG16에서 덴스층을 말한다!(우리껄 써야하므로 False)
    conv_base.trainable = False  # 그 vGG16안에 있는 weight를 학습하지 않겠다는 것이다.(처음부터 하겠다는 뜻- 이거로 고정!)
    model = keras.Sequential()

    # Dense 레이어는 2차원 데이터가 들어간다. but conv 는 4차원이 들어간다. 따라서 중간에 중간다리가 있어야 한다.
    # model.add(keras.layers.Reshape([-1]))
    model.add(conv_base)
    model.add(keras.layers.Flatten())  # 4차원 -> 2차원
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dense(2, activation='softmax'))
    model.summary()

    checkpoint = keras.callbacks.ModelCheckpoint(filepath='model/recycling_{epoch:02d}-{val_loss:.2f}.h5', save_best_only=True)  #모델 학습된 것! 나중에 완제품 단계에서 사용할 것!
    # model.fit단계에서 check

    model.compile(optimizer=keras.optimizers.Adadelta(0.001), #Adadelta 한번 사용해보자!
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    model.summary()
    model.fit(flow_train, epochs=10, batch_size=32, validation_data=flow_test,callbacks=[checkpoint])


    # model.fit_generator은 안쓴다!

    # model.save('model/recycling.h5') #best 모델 저장에 사용!

    # model = keras.models.load_model('model/embedding.h5') #이거로 나중에 불러오기만 할 수 있다!
    # 저장할 때 fit 함수의 변수로 callbacks= checkpoint 를 넣어준다.

    return
def load_model(model_path, data):
    get_test = keras.preprocessing.image.ImageDataGenerator()
    flow_test = get_test.flow_from_directory(data,
                                             batch_size=19,
                                             target_size=[224, 224],
                                             class_mode='sparse')

    x, y = flow_test.next()
    # print(flow_test)
    model=keras.models.load_model(model_path)
    p = model.predict(x)
    p_arg = np.argmax(p, axis=1)
    print(p_arg)
    print(y)
    print(flow_test.classes)
    return


def predict_model(model, img):
    classes = ['glass','metal','plastic']
    img = cv2.imread(img)
    # print(img.shape) #(1062, 743, 3)
    res_img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
    # print(res_img.shape)#(224, 224, 3)
    pre_img = res_img.reshape(1, 224, 224, 3)
    pre_img = pre_img/255.0
    # print(pre_img.shape) #(1, 224, 224, 3)

    my_model = model

    p = my_model.predict(pre_img)
    arg_p = np.argmax(p)
    return classes[arg_p]


if __name__ == '__main__':
    # PreTrained_save_model()
    # load_model('model/recycling_01_3.87.h5', 'plastic_recy2/test')
    # predict_model('model/recycling_01-7.62.h5','data/test_cardboard.JPG')

    k = keras.models.load_model('model/recycling_04-9.96.h5')
    predict_model(k,'data/test_plastic.JPG')