import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd


root=tk.Tk()
root.iconbitmap("favicon.ico")
root.title('Dog breed prediction')
root.geometry('500x720')
root.resizable(False,True)


# Se creaza o functie de incarcare a modelului salvat
def load_model(model_path):
  model=tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
  return model

# crearea functiei de transformare a probabilitatilor de predictie in denumirea rasei
def incarcarea_rase():
    labels_csv = pd.read_csv('rase.csv')
    labels = labels_csv['breed'].to_numpy()
    unique_breeds = np.unique(labels)
    return unique_breeds

def label_rezult(top_rase, top_probabil):
    global rezult_frame
    rezult_frame.destroy()
    rezult_frame = LabelFrame(frame, text='Rezultatele predictei rasei')
    rezult_frame.pack(padx=20, pady=10, fill=X)

    label_rez_1 = tk.Label(rezult_frame, text=f'1. "{top_rase[0].upper()}", probabilitate {top_probabil[0]*100:.2f}%')
    label_rez_2 = tk.Label(rezult_frame, text=f'2. "{top_rase[1].upper()}", probabilitate {top_probabil[1]*100:.2f}%')
    label_rez_3 = tk.Label(rezult_frame, text=f'3. "{top_rase[2].upper()}", probabilitate {top_probabil[2]*100:.2f}%')
    label_rez_4 = tk.Label(rezult_frame, text=f'4. "{top_rase[3].upper()}", probabilitate {top_probabil[3]*100:.2f}%')
    label_rez_5 = tk.Label(rezult_frame, text=f'5. "{top_rase[4].upper()}", probabilitate {top_probabil[4]*100:.2f}%')

    label_rez_1.pack(ipadx=20, fill=X)
    label_rez_2.pack(ipadx=20, fill=X)
    label_rez_3.pack(ipadx=20, fill=X)
    label_rez_4.pack(ipadx=20, fill=X)
    label_rez_5.pack(ipadx=20, fill=X)

# Crearea functiei de procesare a imaginilor
def process_image(image_path, img_size=224):
  """
  Transforma imaginea de pe calea specificata in imagini format Tensor
  """
  image=tf.io.read_file(image_path)
  image=tf.image.decode_jpeg(image, channels=3)
  image=tf.image.convert_image_dtype(image, tf.float32)
  image=tf.image.resize(image, size=[img_size, img_size])
  return image

# Se creaza o functie ce va transforma poza in batch de 32 imagini
def create_data_batches(X):

    data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))
    data_batch = data.map(process_image).batch(32)
    return data_batch


def determinare_rasa():
    custom_image_paths=[]
    custom_image_paths.append(cale_poza)
    custom_data = create_data_batches(custom_image_paths)

    model = load_model("model.h5")

    custom_preds = model.predict(custom_data)

    rase=incarcarea_rase()

    top_5_indexi_pred = np.argsort(custom_preds[0])[::-1][:5]
    top_5_prbabil_pred = custom_preds[0][top_5_indexi_pred]
    top_5_rase_pred=rase[top_5_indexi_pred]

    label_rezult(top_5_rase_pred, top_5_prbabil_pred)

def buton_rasa():
    buton_rasa_frame = LabelFrame(frame, text='Determinați rasa câinelui DVS')
    buton_rasa_frame.pack(padx=20, pady=10, fill=X)

    buton_rasa = tk.Button(buton_rasa_frame, text="Determinare rasă", command=determinare_rasa)
    buton_rasa.pack(padx=20, pady=10, fill=X)

def incarcare_poza():
    global poza_frame, cale_poza
    for img_display in frame.winfo_children():
        img_display.destroy()

    cale_poza = filedialog.askopenfilename( title='Încărcare poza',
                                       filetypes=(("All Files","*"), ('jpeg files', '*.jpeg'), ('jpg files', '*.jpg'),
                                                  ('bitmap files', '*.bmp'),('png files', '*.png')))

    poza_frame = LabelFrame(frame, text='Poza câinelui DVS')
    poza_frame.pack(padx=20, pady=10, fill=X)

    poza = Image.open(cale_poza)
    propor_latim = (300 / float(poza.size[0]))
    propor_inaltime = int((float(poza.size[1]) * float(propor_latim)))
    poza = poza.resize((300, propor_inaltime), Image.ANTIALIAS)
    poza = ImageTk.PhotoImage(poza)

    panel =Label(poza_frame, image=poza)
    panel.image = poza
    panel.pack()
    buton_rasa()

main_frame=tk.Frame(root)
main_frame.pack(expand=1, fill=BOTH)

my_canvas=Canvas(main_frame)
my_canvas.pack(side=LEFT, fill=BOTH, expand=1)

scrollbar = Scrollbar(main_frame, orient=VERTICAL, command = my_canvas.yview)
scrollbar.pack( side = RIGHT, fill=Y)

my_canvas.configure(yscrollcommand=scrollbar.set)
my_canvas.bind('<Configure>', lambda e: my_canvas.configure(scrollregion = my_canvas.bbox("all")))

main_frame1=tk.Frame(my_canvas)
my_canvas.create_window((0,0), window= main_frame1, anchor= "nw", width=490)

incarcare_frame=LabelFrame(main_frame1, text='Încărcați poza câinelui DVS')
incarcare_frame.pack(padx=20, pady=10, fill=X)

buton_incarcare=tk.Button(incarcare_frame, text="Încărcare poză", command=incarcare_poza)
buton_incarcare.pack(padx=20, pady=10, fill=X)

frame = tk.Frame(main_frame1)
frame.pack(fill=X)

rezult_frame = LabelFrame(frame, text='Rezultatele predictei rasei')

root.mainloop()