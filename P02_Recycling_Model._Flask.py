



#파이썬 애니웨어
import sys
import random
import re
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import main
from Day_18_02_Myrecycling import predict_model
import tensorflow.keras as keras

application = Flask(__name__)
@application.route('/')
def index():
    return render_template("temp_for_Day18/home.html")

@application.route('/applyphoto')
def photo_apply():
    location= request.args.get("location")
    cleaness = request.args.get("clean")
    built_in = request.args.get("built")

    print(location, cleaness, built_in)
    return render_template("apply_photo.html")

@application.route("/upload_done", methods=["POST"])
def upload_done():
    uploaded_files = request.files["file"]
    uploaded_files.save("static/img/{}.JPG".format(1))
    return redirect(url_for("predict"))

@application.route('/predict')
def predict():
    a = predict_model(k, 'static/img/1.JPG')
    # a = 'plastic'
    list.append(dict[a])
    return render_template('temp_for_Day18/answer.html', name=a)

@application.route('/recycle_method')
def recycle_method():
    ans = list[0]
    list.remove(list[0])
    return render_template('temp_for_Day18/recycle_method.html', name=ans)



if __name__ == '__main__':
    list = []
    dict = {'plastic':'플라스틱 페트병은, 라벨을 제거한 후, 무색 페트병과 유색 페트병을 나누어서 배출합니다.',
         'metal':'캔은 밟아서 부피를 줄이고, 부탄가스는 구멍을 뚫어 부피를 줄인후, 배출합니다',
         'glass':'유리는 깨지면 병 재사용이 불가능하므로, 깨지지 않도록 모아서 배출합니다.'}

    k = keras.models.load_model('model/recycling_01-9.35.h5')
    application.run(debug=True)  # debug = True를 주면 개발자 서버가 되고 자동으로 수정업데이트가 된다.


# 파이썬 애니웨어에서는 it __name__ 이거는 치지 마라. 거기 안에 알아서 돌려주는 툴이 있다.
# gudwls5863
# kwon6821!@
#
