from flask import Flask,jsonify
app = Flask(__name__)

import pandas as pd 
import numpy as np
import os

df = pd.read_csv('movie_dataset.csv')
df = df.dropna(axis=0, how='any')
df.reset_index(inplace=True)


def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]

def get_data():
    data = df[['genres','keywords','director']] # Filters Numbers
    data['text_data'] = df['title']+" "+df['genres'] +" "+ df['keywords'] + " "+ df['director']
    return data['text_data'].values


def text_victorized(text):
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer()
    cv_fit=cv.fit_transform(text)
    return cv_fit


def cosine_text(cv_fit):
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(cv_fit)

def print_similar(Get_FILM,cosine_text_):
    similar_movies = list(enumerate(cosine_text_[get_index_from_title(Get_FILM)]))
    sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse = True)
    print_similr = []
    for index,movie in enumerate(sorted_similar_movies):
        if get_title_from_index(movie[0]) != Get_FILM:
            print(get_title_from_index(movie[0]),'-Similarty:',round(movie[1]*100,3),'%')
            print_similr.append({'title':get_title_from_index(movie[0]),'similarty':round(movie[1]*100,3)})
        if index == 10 :
            break
    return print_similr




text_representaion = get_data()
text_victorized_ = text_victorized(text_representaion)
cosine_text_ = cosine_text(text_victorized_)   


@app.route('/movie/<string:Get_FILM>',methods=['GET'])
def index(Get_FILM):
	try:
		data = print_similar(Get_FILM.title(),cosine_text_)
		return jsonify({'data':data}),200
	except:
		return jsonify({'error':'Please Try Anthor Name'})

if __name__ == '__main__':
	app.run(debug=True)