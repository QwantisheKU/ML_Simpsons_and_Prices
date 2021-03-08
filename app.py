from flask import Flask, redirect, url_for, request, render_template
import numpy as np
import os
from werkzeug.utils import secure_filename

from tensorflow import keras
import tensorflow as tf
import pickle
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

try: 
    import shutil 
    shutil.rmtree('uploaded / image') 
    print() 
except: 
    pass
  
app = Flask(__name__)

#Пути к static и uploaded/image
BASE_DIR = os.path.dirname(__file__)
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploaded/image/')
PATH_TO_STATIC = os.path.join(BASE_DIR, 'static/')

model = tf.keras.models.load_model('simpsons.h5') 
model_prices = pickle.load(open('sale.pkl', 'rb'))

#Назначение меток и имен 
names_dict = {0: 'Абрахам Симпсон',
 1: 'Агнес Скиннер',
 2: 'Апу Нахасапимапетилон',
 3: 'Барни Гамбл',
 4: 'Барт Симпсон',
 5: 'Карл Карлсон',
 6: 'Чарльз Монтгомери Бёрнс',
 7: 'Шериф Виггам',
 8: 'Клетус Спаклер',
 9: 'Продавец комиксов',
 10: 'Диско Стю',
 11: 'Эдна Крабаппл',
 12: 'Жирный Тони',
 13: 'Гил Гундерсон',
 14: 'Садовник Вилли',
 15: 'Гомер Симпсон',
 16: 'Кент Брокман',
 17: 'Клоун Красти',
 18: 'Ленни Леопард',
 19: 'Лайнел Хатц',
 20: 'Лиза Симпсон',
 21: 'Мэгги Симпсон',
 22: 'Мардж Симпсон',
 23: 'Мартин Принц',
 24: 'Мэр Куимби',
 25: 'Милхаус Ван Хутен',
 26: 'Мисс Хувер',
 27: 'Мо Сизлак',
 28: 'Нед Фландерс',
 29: 'Нельсон Манц',
 30: 'Отто Манн',
 31: 'Пэтти Бувье',
 32: 'Директор Скиннер',
 33: 'Профессор Фринк',
 34: 'Райнер Вульфкасл',
 35: 'Ральф Виггам',
 36: 'Сельма Бувье',
 37: 'Сайдшоу Боб',
 38: 'Сайдшоу Мел',
 39: 'Змей Джейлбёрд',
 40: 'Трой Макклюр',
 41: 'Вэйлон Смитерс'}

names_arr = []
for i in names_dict.values():
    names_arr.append(i)

@app.route('/')
def home():
    return render_template('main.html')

#Содание генератора картинок + прогноз
def model_data(): 
    #Создание генератора под keras модель
    test_datagen = ImageDataGenerator(rescale = 1./255) 
    test_dir = os.path.join(BASE_DIR, 'uploaded/')
    test_generator = test_datagen.flow_from_directory( 
            test_dir, 
            target_size =(224, 224), 
            color_mode ="rgb", 
            shuffle = True, 
            class_mode ='categorical', 
            batch_size = 1)
             
    #Прогнозы от модели
    pred = model.predict_generator(test_generator) 
    classes = model.predict_proba(test_generator)
    final = names_arr[np.argmax(pred)]
    global prob
    prob = f"Вероятность: ~{int(max(classes[0])*100)}% "
    return f"{final}" 

#Функиця для прогноза цены на дом
@app.route('/predict_prices', methods=['GET','POST'])
def predict_prices():
    if request.method == 'POST':
        features = [int(x) for x in request.form.values()]
        final_features = [np.array(features)]
        pred = model_prices.predict(final_features)
        return render_template('prices.html', data=f"Price of the house is about {round(pred[0], 2)}$")
    else:
        return render_template('prices.html')

#Функция для классификация персонажа из Симпсонов
@app.route('/predict_simpsons', methods=['GET','POST'])
def predict_simpsons():
    filelist = [ f for f in os.listdir(PATH_TO_STATIC)]
    for f in filelist:
        os.remove(os.path.join(PATH_TO_STATIC, f))
    if request.method == 'POST':
        filelist = [ f for f in os.listdir(app.config['UPLOAD_FOLDER'])]
        for f in filelist:
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f)) 
        f = request.files['file']
        
        image_for_upload = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        f.save(image_for_upload) 
        val = model_data() 

        global image
        image = image_for_upload
        shutil.copy(image, PATH_TO_STATIC)
        final_image = os.listdir(PATH_TO_STATIC)[0]
        result = render_template('index.html', ss = val, prob=prob, image=final_image)
        return result
    else:
        return render_template('index.html')

if __name__=='__main__':
    app.run(debug=True)