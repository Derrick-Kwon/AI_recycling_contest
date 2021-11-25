#http://web.cecs.pdx.edu/~singh/rcyc-web/dataset.html 포틀랜드주립대학 dataset
#taco: https://github.com/pedropro/TACO
#trash_classification: https://github.com/gibiee/Trash_Classification
#url 받아오는 법! #유튜브! / html 카메라사용 검색 ㄱㄱ
#html로 인터페이스 꾸미기!
##############################################
#이미지 import해서 4차원으로 만들고 바로 넣는다.
#질문1: 임의의 사진이 있을때 이걸 모델에 넣어서 보기 위해선 어떻게 해야 하는지?? @

#질문2: 모델 callbacks로 체크포인트 만드려고, validation 을 만들었는데, cnn 에선 validation 을 어떻게 넣어야 하는가??? @

#질문3:  가끔씩 보면 weight값을 리턴해와서 뽑아서 보는 경우도 있는데 어떤 모델일 때 이렇게 하는지? ?

#질문4:
#해결해야하는 문제1 : 모델 저장해서 나의 임의의 사진을 돌려서 결괏값(무엇인지) 리턴받기 @
#해결해야하는 문제2 : flask와 연결해서 모델의 임의의 사진 결과값 얻은것을 웹페이지에 보내기. @
#해결해야하는 문제3 : 카메라와 html, python 연결해서 파이썬으로 사진 받아서 웹페이지에 저장하기
#해결해야 하는 문제4 : 문제 1, 2, 3 합쳐서 사진을 찍어서 저장 -> python 으로 가져오기->모델돌리기->사진 리턴 및 결괏값과 함께 다시 웹페이지로 보내기

# +a 전이학습모델로 모델을 업데이트하기: 앱에서 찍은 사진으로 끊임없이 업데이트 시키는 모델. train set에 내가 찍어서 돌렸던 사진을 추가!
#현재는 VGG16과 Dense를 연결할때 1차원으로 바꿔줘서 이미지 보강 기법을 사용하지 못하였지만,
#여력이 된다면 GPU를 이용해서 이미지 보강도 해보자!



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

    flow_train = gen_train.flow_from_directory('plastic_recy2/train',  # ctrl써서 들어가본다. 그리고 안에 필요한거를 채운다.
                                               target_size=[224, 224],
                                               class_mode='sparse')  # categorical, binary, sparse' 기본 코드 xx

    get_test = keras.preprocessing.image.ImageDataGenerator()
    flow_test = get_test.flow_from_directory('plastic_recy2/test',
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
    model.add(keras.layers.Dense(3, activation='softmax'))
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


def predict_model(model_path, img):
    classes = ['glass','metal','plastic']
    img = cv2.imread(img)
    # print(img.shape) #(1062, 743, 3)
    res_img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
    # print(res_img.shape)#(224, 224, 3)
    pre_img = res_img.reshape(1, 224, 224, 3)
    # print(pre_img.shape) #(1, 224, 224, 3)

    model = keras.models.load_model(model_path)

    p = model.predict(pre_img)
    arg_p = np.argmax(p)
    print(classes[arg_p])
    exit()

    return
# PreTrained_save_model()
# load_model('model/recycling_01_3.87.h5', 'plastic_recy2/test')
# predict_model('model/recycling_01-7.62.h5','data/test_cardboard.JPG')
predict_model('model/recycling_01-9.35.h5','data/test_plastic.JPG')