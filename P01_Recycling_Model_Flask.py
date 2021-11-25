#Day_15_02_Flask #인터페이스인듯 하다.
# https://tutorial.djangogirls.org/ko/ #튜토리얼로 장고걸스 여기서 인터페이스를 만들어보자

# static, templates 두개의 폴더를 만든다(폴더이름 틀리면 ㄴㄴ)
# static은 정적: 바뀌지 않는데이터를 여기 넣는다.
# templates: html5 파일을 여기 넣는다.
# templates -> html 파일 만들어봐라!


#파이썬 애니웨어
import sys
import random
import re
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import P01_Recycling_Model
from P01_Recycling_Model import predict_model

application = Flask(__name__)
@application.route('/')
def index():
    return render_template("Recycling.html")

@application.route('/apply')
def picture():
    return render_template("take a picture.html")

@application.route('/predict')
def predict():
    # a = 10
    a = predict_model('model/recycling_01-9.35.h5', 'static/img/1.JPG')
    return render_template('practice_01.html', name=a)

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


@application.route('/save', methods=['POST']) #파일 주고받는거는 get방식과 post 방식이 있다
def save_image():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(os.path.join('static', filename))
        # 저장할 경로 + 파일명
        f.save(os.path.join('static', secure_filename((f.filename))))
        return 'uploads 디렉토리 -> 파일 업로드 성공!'
    pass



if __name__ == '__main__':
    application.run(debug=True)  # debug = True를 주면 개발자 서버가 되고 자동으로 수정업데이트가 된다.


# 파이썬 애니웨어에서는 it __name__ 이거는 치지 마라. 거기 안에 알아서 돌려주는 툴이 있다.
# gudwls5863
# kwon6821!@
#
