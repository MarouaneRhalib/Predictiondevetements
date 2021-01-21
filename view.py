from flask import render_template, Flask, url_for
from flask import request, redirect
import os
from keras_preprocessing.image import load_img, img_to_array

from m1.lab3 import ModelDeploy
Upload_F = 'uploads'
app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = Upload_F

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":

        if request.files:
            image = request.files["image"]
            file_path = os.path.join(app.config["IMAGE_UPLOADS"], image.filename)
            image.save(file_path)
            print("image saved")
            print(file_path)
            PredictionResult = ModelDeploy.predict(file_path)
            print("success")
            return render_template('index.html', PredictionResult = PredictionResult, user_image = file_path)
    return render_template("index.html")





if __name__=='__main__':
    app.run(debug= True)
