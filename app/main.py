import streamlit as st
from keras import layers
from keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, 
                          Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,
                          Dropout)
from keras.models import Model, load_model
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from keras.initializers import glorot_uniform
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img,img_to_array
from bing_image_downloader import downloader
from streamlit import caching
from PIL import Image as imagine
from PIL import Image
import librosa
import numpy as np
import librosa.display
from pydub import AudioSegment
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import wave
import contextlib
import os
import random


IMAGE_SAVE_LOCATION = 'D:/CNN/app/melspectrogram.png'
CONVERTED_SONG_LOCATION = "D:/CNN/app/music_file.wav"
EXTRACTED_SONG_LOCATION = "D:/CNN/app/extracted.wav"
PHOTOS_FOLDER = "D:/CNN/app/photos"
APP_FOLDER = "D:/CNN/app"

class_labels = ['blues',
 'classical',
 'country',
 'disco',
 'hiphop',
 'metal',
 'pop',
 'reggae',
 'rock']

scores = {'blues':0,
 'classical':0,
 'country':0,
 'disco':0,
 'hiphop':0,
 'metal':0,
 'pop':0,
 'reggae':0,
 'rock':0}

rec = {'blues':["Mighty Mo Rodgers - Picasso Blue", "Pete Gage - Blues Has Got Me", "Muddy Waters - Hoochie Coochie Man", "Oli Brown - I Love You More You'll Ever Know", "Guy Davis - Loneliest Road That I Know"],
 'classical':["Moonlight - Ludwig Van Beethoven", "Messiah - George Frideric Handel", "Eine kleine Nachtmusik - Wolfgang Amadeus Mozart", "The Blue Danube - Johann Strauss II", "Symphony No 5 - Ludwig van Beethoven"],
 'country':["I Walk the Line - Johnny Cash", "Friends in Low Places - Garth Brooks", "Choices - George Jones", "Concrete Angel - Martina McBride", "Where Were You - Alan Jackson"],
 'disco':["Donna Summer - I Feel Love", "Silver Convention - Fly Robin Fly", "Gloria Gaynor - I Will Survive", "Sister Sledge - We Are Family", "Tavares - Heaven Must Be Missing an Angel"],
 'hiphop':["Geto Boys - Mind Playing Tricks on Me", "Dr. Dre - Nuthin' but a 'G' Thang", "Public Enemy - Fight the Power", "N.W.A - Straight Outta Compton", "Public Enemy - Rebel Without a Pause"],
 'metal':["System Of A Down - Chop Suey!", "Deftones – Change", "Slipknot – Duality", "Mastodon – Blood And Thunder", "Rammstein – Sonne"],
 'pop':["Ed Sheeran - Shape of You","Carly Rae Jepsen - Call Me Maybe", "Lewis Capaldi - Before You Go", "BTS (방탄소년단) - Dynamite", "The Chainsmokers - Closer", "Alessia Cara - Scars To Your Beautiful"],
 'reggae':["One Love – Bob Marley And The Wailers", "The Tide Is High – The Paragons", "Bam Bam – Sister Nancy", "Hold Me Tight – Johnny Nash", "Many Rivers To Cross – Jimmy Cliff"],
 'rock':["Purple Haze - Jimi Hendrix", "Whole Lotta Love - Led Zeppelin", "Sympathy for the Devil - The Rolling Stones", "Under Pressure - Queen & David Bowie", "Comfortably Numb - Pink Floyd"]}


def GenreModel(input_shape = (288,432,3),classes=9):
  np.random.seed(9)
  X_input = Input(input_shape)

  X = Conv2D(8,kernel_size=(3,3),strides=(1,1),kernel_initializer = glorot_uniform(seed=9))(X_input)
  X = BatchNormalization(axis=3)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2))(X)
  
  X = Conv2D(16,kernel_size=(3,3),strides = (1,1),kernel_initializer=glorot_uniform(seed=9))(X)
  X = BatchNormalization(axis=3)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2))(X)
  
  X = Conv2D(32,kernel_size=(3,3),strides = (1,1),kernel_initializer = glorot_uniform(seed=9))(X)
  X = BatchNormalization(axis=3)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2))(X)

  X = Conv2D(64,kernel_size=(3,3),strides=(1,1),kernel_initializer=glorot_uniform(seed=9))(X)
  X = BatchNormalization(axis=-1)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2,2))(X)

  
  X = Flatten()(X)

  X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=9))(X)

  model = Model(inputs=X_input,outputs=X,name='GenreModel')

  return model


def convert_mp3_to_wav(music_file):
    sound = AudioSegment.from_mp3(music_file)
    sound.export(CONVERTED_SONG_LOCATION,format="wav")

def extract_relevant(wav_file,t1,t2):
    wav = AudioSegment.from_wav(wav_file)
    wav = wav[1000*t1:1000*t2]
    wav.export(EXTRACTED_SONG_LOCATION,format='wav')

def create_melspectrogram(wav_file):
    y,sr = librosa.load(wav_file,duration=5)
    mels = librosa.feature.melspectrogram(y=y,sr=sr)

    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.figsize"] = [4.32, 2.88]
    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    p = plt.imshow(librosa.power_to_db(mels,ref=np.max))
    plt.savefig(IMAGE_SAVE_LOCATION)

def scr(s):
    scores.update({s: scores.get(s)+1})

def song_type(song):
    if song == 0:
        scr("blues")
    elif song == 1:
        scr("classical")
    elif song == 2:
        scr("country")
    elif song == 3:
        scr("disco")
    elif song == 4:
        scr("hiphop")
    elif song == 5:
        scr("metal")
    elif song == 6:
        scr("pop")
    elif song == 7:
        scr("reggae")
    elif song == 8:
        scr("rock")

def song_seg():
    test = 0
    song = ''
    for i in class_labels:
        if scores.get(i) > test:
            test = scores.get(i)

    for i in class_labels:
        if scores.get(i) == test:
            song = i
    return song

def download_image():
    filename = file.name
    filename = str.split(filename,".")[0]
    downloader.download(filename, limit=1,  output_dir=PHOTOS_FOLDER, adult_filter_off=True, force_replace=False, timeout=60)
    return filename

def download_image_demo(filename):
    downloader.download(filename, limit=1,  output_dir=PHOTOS_FOLDER, adult_filter_off=True, force_replace=False, timeout=60)
  
def predict(image_data,model):

    image = img_to_array(image_data)
    image = np.reshape(image,(1,288,432,3))
    prediction = model.predict(image/255)
    prediction = prediction.reshape((9,)) 
    class_label = np.argmax(prediction)
    
    return class_label,prediction

def check_photo(songn):
    if os.path.exists(PHOTOS_FOLDER+"/"+songn+"/Image_1.jpg"):
        song_photo = imagine.open(PHOTOS_FOLDER+"/"+songn+"/Image_1.jpg")
    elif os.path.exists(PHOTOS_FOLDER+"/"+songn+"/Image_1.png"):
        song_photo = imagine.open(PHOTOS_FOLDER+"/"+songn+"/Image_1.png")
    elif os.path.exists(PHOTOS_FOLDER+"/"+songn+"/Image_1.jpeg"):
        song_photo = imagine.open(PHOTOS_FOLDER+"/"+songn+"/Image_1.jpeg")
    else:
        song_photo = imagine.open(APP_FOLDER+"/Default_photo/placeholder-image.png")
    return song_photo

def show_output(songname):
    convert_mp3_to_wav(APP_FOLDER+"/preload/"+songname + ".mp3")
    fname = CONVERTED_SONG_LOCATION
    with contextlib.closing(wave.open(fname,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    duration = int(duration/5)
    for i in range(0,duration):
        extract_relevant(CONVERTED_SONG_LOCATION, i*5, (i+1)*5 )
        create_melspectrogram(EXTRACTED_SONG_LOCATION) 
        image_data = load_img(IMAGE_SAVE_LOCATION, target_size=(288,432,3))
        class_label,prediction = predict(image_data, model)

        prediction = prediction.reshape((9,)) 
        song_type(class_label)


    updated_scores = scores.copy()
    for i in class_labels:
        updated_scores.update({i: updated_scores.get(i)/duration})
    
    st.write("## Genul cantecului prezis: "+song_seg())
    
    download_image_demo(songname)
    st.sidebar.write(songname)
    st.sidebar.image(check_photo(songname),use_column_width=True)
    st.sidebar.audio(APP_FOLDER+"/preload/"+songname+".mp3","audio/mp3")

    color_data = [1,2,3,4,5,6,7,8,9]
    my_cmap = cm.get_cmap('jet')
    my_norm = Normalize(vmin=0, vmax=9)

    fig,ax= plt.subplots(figsize=(8,6))
    ax.bar(x = class_labels, height = updated_scores.values(), color = my_cmap(my_norm(color_data)))
    plt.xticks(rotation=0)
    ax.set_title(f" {duration} de segmente extrase din probabilitatea de distributie in divizii de 5 secunde.")
    plt.show()
    st.pyplot(fig)


    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write("## Diagrama circulara cu probabilitatea tipului de cantec")
    for i in class_labels:
        updated_scores.update({i: updated_scores.get(i)/duration*100})

    nume = [key for key,value in updated_scores.items() if value!=0]
    valori = [value for value in updated_scores.values() if value!=0]
    fig1, ax1 = plt.subplots()
    ax1.pie(valori, autopct=lambda p: '{:.1f}%'.format(round(p,2)) if p > 0 else '',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.legend(labels = nume, loc = 'center right', bbox_to_anchor=(1.4,0.5))
    plt.show()
    st.pyplot(fig1)

    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write("## Recomandari de melodii similare")
    col1, col2, col3 = st.columns(3)
    
    val = random.choice(rec.get(song_seg()))
    download_image_demo(val)
    col1.write(val)
    col1.image(check_photo(val), use_column_width=True)
    rec.get(song_seg()).remove(val)

    val = random.choice(rec.get(song_seg()))
    download_image_demo(val)
    col2.write(val)
    col2.image(check_photo(val), use_column_width=True)
    rec.get(song_seg()).remove(val)

    val = random.choice(rec.get(song_seg()))
    download_image_demo(val)
    col3.write(val)
    col3.image(check_photo(val), use_column_width=True)
    rec.get(song_seg()).remove(val)
    
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write(f"## Predictiile divizilor de 5 secunde. Numar de segmente totale: {duration}")
    st.write(scores)

st.write(""" # Sistem inteligent de recunoastere a genului muzical""")

st.markdown(
    f'''
        <style>
            .sidebar .sidebar-content {{
                width: 390px;
            }}
        </style>
    ''',
    unsafe_allow_html=True
)


file = st.sidebar.file_uploader("Incarcati un fisier mp3 sau folositi un exemplu preincarcat de mai jos:",type=["mp3"])

model = GenreModel(input_shape=(288,432,3),classes=9)
model.load_weights(APP_FOLDER+'/Music_CNN.h5')


demo = st.sidebar.checkbox("Exemple preincarcate (cate 1 pentru fiecare gen)")

if(demo):
  
      
    song = st.sidebar.radio("Ce melodie vrei sa verifici?",("BB King - The Thrill Is Gone","Johann Sebastian Bach - Toccata and Fugue in D Minor",
    "Dolly Parton - Jolene","ABBA - Dancing Queen","50 Cent - In Da Club", "Metallica - Enter Sandman", 
    "Shawn Mendes - Señorita", "Bob Marley & The Wailers - Three Little Birds", "AC-DC - You Shook Me All Night Long"))

    if(song=="BB King - The Thrill Is Gone"):
        show_output("BB King - The Thrill Is Gone") 
    if(song=="Johann Sebastian Bach - Toccata and Fugue in D Minor"):
        show_output("Johann Sebastian Bach - Toccata and Fugue in D Minor")
    if(song=="Dolly Parton - Jolene"):
        show_output("Dolly Parton - Jolene")
    if(song=="ABBA - Dancing Queen"):
        show_output("ABBA - Dancing Queen")
    if(song=="50 Cent - In Da Club"):
        show_output("50 Cent - In Da Club")
    if(song=="Metallica - Enter Sandman"):
        show_output("Metallica - Enter Sandman")
    if(song=="Shawn Mendes - Señorita"):
        show_output("Shawn Mendes - Señorita")
    if(song=="Bob Marley & The Wailers - Three Little Birds"):
        show_output("Bob Marley & The Wailers - Three Little Birds")
    if(song=="AC-DC - You Shook Me All Night Long"):
        show_output("AC-DC - You Shook Me All Night Long")

elif file is None:
    st.text("Incarcati un fisier mp3.")
else:
    convert_mp3_to_wav(file)

    fname = CONVERTED_SONG_LOCATION
    with contextlib.closing(wave.open(fname,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    duration = int(duration/5)
    for i in range(0,duration):
        extract_relevant(CONVERTED_SONG_LOCATION, i*5, (i+1)*5 )
        create_melspectrogram(EXTRACTED_SONG_LOCATION) 
        image_data = load_img(IMAGE_SAVE_LOCATION, target_size=(288,432,3))
        class_label,prediction = predict(image_data, model)

        prediction = prediction.reshape((9,)) 
        song_type(class_label)


    updated_scores = scores.copy()
    for i in class_labels:
        updated_scores.update({i: updated_scores.get(i)/duration})
    
    
    st.write("## Genul cantecului prezis: "+song_seg())
    
    filename = download_image()
    st.sidebar.write(filename)
    st.sidebar.image(check_photo(filename),use_column_width=True)
    st.sidebar.audio(file,"audio/mp3")
    
    
    
    color_data = [1,2,3,4,5,6,7,8,9]
    my_cmap = cm.get_cmap('jet')
    my_norm = Normalize(vmin=0, vmax=9)

    fig,ax= plt.subplots(figsize=(8,6))
    ax.bar(x = class_labels, height = updated_scores.values(), color = my_cmap(my_norm(color_data)))
    plt.xticks(rotation=0)
    ax.set_title(f" {duration} de segmente extrase din probabilitatea de distributie in divizii de 5 secunde.")
    plt.show()
    st.pyplot(fig)


    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write("## Diagrama circulara cu probabilitatea tipului de cantec")
    for i in class_labels:
        updated_scores.update({i: updated_scores.get(i)/duration*100})

    nume = [key for key,value in updated_scores.items() if value!=0]
    valori = [value for value in updated_scores.values() if value!=0]
    fig1, ax1 = plt.subplots()
    ax1.pie(valori, autopct=lambda p: '{:.1f}%'.format(round(p,2)) if p > 0 else '',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.legend(labels = nume, loc = 'center right', bbox_to_anchor=(1.4,0.5))
    plt.show()
    st.pyplot(fig1)

    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write("## Recomandari de melodii similare")
    col1, col2, col3 = st.columns(3)
    
    val = random.choice(rec.get(song_seg()))
    col1.write(val)
    download_image_demo(val)
    col1.image(check_photo(val), use_column_width=True)
    rec.get(song_seg()).remove(val)

    val = random.choice(rec.get(song_seg()))
    col2.write(val)
    download_image_demo(val)
    col2.image(check_photo(val), use_column_width=True)
    rec.get(song_seg()).remove(val)

    val = random.choice(rec.get(song_seg()))
    col3.write(val)
    download_image_demo(val)
    col3.image(check_photo(val), use_column_width=True)
    rec.get(song_seg()).remove(val)

    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write(f"## Predictiile divizilor de 5 secunde. Numar de segmente totale: {duration}")
    st.write(scores)

