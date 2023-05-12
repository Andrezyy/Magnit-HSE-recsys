import hashlib
import os
from pathlib import Path
import pandas as pd
from loguru import logger
from flask import Flask, render_template, request, redirect, url_for
from .config import *

# Создаем логгер и отправляем информацию о запуске
# Важно: логгер в Flask написан на logging, а не loguru,
# времени не было их подружить, так что тут можно пересоздать 
# logger из logging
logger.add(LOG_FOLDER + "log.log")
logger.info("Наш запуск")

# Создаем сервер и убираем кодирование ответа
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

#------------------МОЕ----------------------
import numpy as np
import pandas as pd
import fastai
from fastai.collab import load_learner, CollabDataLoaders, collab_learner

# Функиция для получения n рекомендаций для UID
def get_k_recs(model, uid, n = 10, known_jokes = None):
    n_jokes = 100
    joke_ids = np.arange(1,n_jokes+1).reshape(-1,1)
    user = np.array([uid]*n_jokes).reshape(-1,1)
    data = np.concatenate([user, joke_ids], axis = 1)
    data = pd.DataFrame(data = data, columns = ['UID','JID'])
    
    preds,_ = model.get_preds(dl = model.dls.test_dl(data))
    preds = preds.unsqueeze(1).numpy()
    
    preds = np.concatenate([joke_ids, preds],axis = -1)
    preds = preds[preds[:, 1].argsort()][::-1]
    top_rmse = {preds[0,0]: preds[0, 1]}
    preds = preds[:, 0]
    
    if known_jokes is None:
        return [top_rmse, preds[:n+1]]
    else:
        preds = preds[~np.isin(preds, known_jokes)]
        return [top_rmse, preds[:n+1]]

# Функция для получения 10 шуток для которого не было в системе UID
# Из-за отсутствия какой либо информации и вкусах пользователя
# не может идти речь о правильном ранжировании
# в качестве лучшей шутки и rmse берется случайная шутка из рекомендации и средний рейтинг посчитанный по тренировочным данным
def cold_start():
    recs = np.array([])
    for genre in joke_genres:
        j = jokes[jokes[genre] == 1].JokeId.values
        j = j[~np.isin(j, recs)]
        j = np.random.choice(j, 2)
        recs = np.concatenate([recs, j])
    return [{np.random.choice(recs, 1)[0]: 0.8817}, recs]
        
    


# Подгружаем данные о шутках и данные
jokes = pd.read_csv(JOKES_DATA)
joke_genres = ['Wordplay', 'Observational', 'Character_based','Satire', 'Absurd', 'Dark']
data = pd.read_csv(TRAIN_DATA)
# Подгружаем модель
dataloader = CollabDataLoaders.from_df(data, user_name='UID', item_name='JID', rating_name= "Rating", bs=2**14, valid_pct=0.1)
model = collab_learner(dataloader, n_factors=370, y_range=[-10.0 ,10.0])
model.load(MODEL)
#--------------------------------------------



@app.route("/<task>")
def main(task: str):
    """
    Эта функция вызывается при вызове любой страницы, 
    для которой нет отдельной реализации

    Пример отдельной реализации: add_data
    
    Параметры:
    ----------
    task: str
        имя вызываемой страницы, для API сделаем это и заданием для сервера
    """
    return render_template('index.html', task=task)

@app.route("/add_data", methods=['POST'])
def upload_file():
    """
    Страница на которую перебросит форма из main 
    Здесь происходит загрузка файла на сервер
    """
    def allowed_file(filename):
        """ Проверяем допустимо ли расширение загружаемого файла """
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    # Копируем шаблон ответа для сервера и устанавливаем выполняемую задачу
    answer = ANSWER.copy()
    answer['Задача'] = 'add_data'

    # Проверяем наличие файла в запросе
    if 'file' not in request.files:
        answer['Сообщение'] = 'Нет файла'
        return answer
    file = request.files['file']
    
    # Проверяем что путь к файлу не пуст
    if file.filename == '':
        answer['Сообщение'] = 'Файл не выбран'
        return answer
    
    # Загружаем
    if file and allowed_file(file.filename):
        file_type = file.filename[file.filename.find('.'):]
        filename = 'input'
        #filename = hashlib.md5(file.filename.encode()).hexdigest() 
        file.save(
            os.path.join(
                DATA_FOLDER, 
                filename + file_type
                )
            )
        answer['Сообщение'] = 'Файл успешно загружен!'
        answer['Успех'] = True
        answer['Путь'] = filename
        return answer
    else:
        answer['Сообщение'] = 'Файл не загружен'
        return answer
        
@app.route("/show_data", methods=['GET'])
def show_file():
    """
    Страница выводящая содержимое файла
    """
    # Копируем шаблон ответа для сервера и устанавливаем выполняемую задачу
    
    answer = ANSWER.copy()
    answer['Задача'] = 'show_file'

    # Проверяем, что указано имя файла
    if 'path' not in request.args:
        answer['Сообщение'] = 'Не указан путь файла'
        return answer
    file = request.args.get('path') 
    
    # Проверяем, что указан тип файла
    if 'type' not in request.args:
        answer['Сообщение'] = 'Не указан тип файла'
        return answer
    type = request.args.get('type')

    file_path = os.path.join(DATA_FOLDER, file + '.' + type)

    # Проверяем, что файл есть
    if not os.path.exists(file_path):
        answer['Сообщение'] = 'Файл не существует'
        return answer

    answer['Сообщение'] = 'Файл успешно загружен!'
    answer['Успех'] = True
    
    # Приводим данные в нужный вид
    if type == 'csv':
        answer['Данные'] = pd.read_csv(file_path).to_dict()
        return answer
    else:
        answer['Данные'] = 'Не поддерживаемы тип'
        return answer
    
# GET запрос для получения рекомендаций для UID из input.csv
# На выходе будет файл output.csv
# Вида: UID, [{top 1 rec: rating},[top 10 recs]]

# Функция если входные данные (уже в виде датафрейма) будут формата: index, UID 
@app.route("/get_recs")
def get_recs():
    
    users = pd.read_csv(DATA_FOLDER + 'input.csv')
    recs = []
    for i, row in users.iterrows():
        uid = row.UID
        with model.no_bar(), model.no_logging():
            # Находим шутки которые пользователь уже видел
            known_jokes = data[data.UID == uid].JID.values
            # Выбираем из них половину а другую половину забываем. То есть некоторые прочитанные шутки могут попасться снова
            known_jokes = np.random.choice(known_jokes, size = int(np.round(len(known_jokes)/2)))
            # Это своеобразный трейдоф между novelty и точностью MAP@10, так как рейтинги знакомых шуток система знает хорошо 
            # Да и логику рекомендательных систем это никак не нарушает, поскольку нет ничего такого, чтобы порекомендовать
            # что то такое, что пользователь уже видел или пробовал. Главное чтобы ему это понравилось!
            
            # Я знаю, что организаторы использовали только jester-data-1
            # Значит у всех известных UID будет информация о 36 шутках и больше
            # Поэтому холодный старт тут простой, так как у пользователя может быть либо >= 36 шуток, либо 0
            if len(known_jokes) == 0:
                rec = cold_start()
            else:
                rec = get_k_recs(model, uid, n = 10, known_jokes = known_jokes)
            recs.append(rec)
        
    users['Recs'] = recs
    users.to_csv(DATA_FOLDER + 'output.csv', index = False)
    
    return 'Для того, чтобы взглянуть на результат перейдите по следующей ссылке: /show_data?path=output&type=csv'
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    