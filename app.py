from multiprocessing import allow_connection_pickling
import os
import cv2
import time
import requests
import tempfile

from PIL import Image 
from flask import Flask, redirect, url_for, request
from main_app import svm_model, is_jpg
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
@app.route('/', methods=['GET'])
def welcome():
    return "API SIGEOO"

@app.route('/api/svm/', methods=['GET', 'POST'])
def svmpath(): 
    if request.method == "POST":
        
        print(request)
        print(request.values)
        
        if 'data_gambar' not in request.files:
            result = {
                "state": False,
                "message": "Data gambar kosong",
                "data": 'null'
            }
            return result
        
        print(request.files['data_gambar'])
        file = request.files['data_gambar']
        
        if file.filename == '':
            result = {
                "state": False,
                "message": "Data gambar kosong",
                "data": 'null'
            }
            return result
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(os.getcwd(), 'images', filename)
            file.save(filepath)
            
            file_open = Image.open(filepath)
            print(file_open)
            
            if os.path.exists(filepath):                
                prediction = svm_model(filename)
                return prediction
            else:
                result = {
                    "state": False,
                    "message": "Dataset tidak ada",
                    "data": 'null'
                }
            return result
            
            # if is_jpg(filename):
            #     if os.path.exists(os.path.join(os.getcwd(), 'images', filename)):                
            #         prediction = svm_model(filename)
            #         return prediction
            #     else:
            #         result = {
            #             "state": False,
            #             "message": "Dataset tidak ada",
            #             "data": 'null'
            #         }
            #     return result
            # else:
            #     path_image = time.strftime("%Y%m%d-%H%M%S.jpg")
            #     file_img = os.path.join(os.getcwd(), 'images', filename)
            #     file_png = Image.open(file_img)
                
            #     print(file_png)
                
            #     convert = file_png.convert('RGB')
            #     convert.save(os.path.join(os.getcwd(), 'images', path_image))
            #     os.remove(file_img)
                
            #     prediction = svm_model(path_image)
            #     return prediction
            
        # data = ''
        # prediction = ''
        
        # # data = request.form.get("data_gambar")        
        # # print("\nData request: " + data)
        
        # file = request.files['data_gambar']
        # print("\nFile request: {}".format(file))
        
        # # print(os.path.exists(os.path.join(os.getcwd(), 'images', data)))
                
        # if file == '':
        #     result = {
        #         "state": False,
        #         "message": "Data gambar kosong",
        #         "data": 'null'
        #     }
        #     return result
        # else:
        #     filename = secure_filename(file.filename)
        #     file.save(os.path.join(os.getcwd(), 'images', filename))
            
        #     print(filename)
            
        #     if is_jpg(filename):
        #         if os.path.exists(os.path.join(os.getcwd(), 'images', filename)):                
        #             prediction = svm_model(filename)
        #             return prediction
        #         else:
        #             result = {
        #                 "state": False,
        #                 "message": "Dataset tidak ada",
        #                 "data": 'null'
        #             }
        #         return result
        #     else:
                
        #         path_image = time.strftime("%Y%m%d-%H%M%S.jpg")
        #         file_img = os.path.join(os.getcwd(), 'images', filename)
        #         file_png = Image.open(file_img.strip())
                
        #         print(file_png)
                
        #         convert = file_png.convert('RGB')
        #         convert.save(os.path.join(os.getcwd(), 'images', path_image))
        #         os.remove(file_img)
                
        #         prediction = svm_model(path_image)
        #         return prediction
            
    elif request.method == "GET":
        return "API SIGEOO"
            

if __name__ == '__main__':
    app.run(host='192.168.191.93', port=8089, debug=True)
    # app.run(debug=True)
