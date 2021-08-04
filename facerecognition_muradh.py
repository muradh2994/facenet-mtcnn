#Face Recognition using Facenet model and MTCNN detector
#Pre-requisite: pip install mtcnn

from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from os import listdir
from os.path import isdir
from matplotlib import pyplot
from numpy import load
from numpy import expand_dims
from numpy import asarray
from keras.models import load_model
import os
import numpy as np
from numpy import savez_compressed

def extract_face(filename, required_size=(160,160)):
    image = Image.open(filename)
    image = image.convert("RGB")
    pixels = np.array(image)  # PIL.Image' has no attribute 'asarray
    #print("image in array", pixels)
    #open detectort
    detector = MTCNN()
    person = detector.detect_faces(pixels)
    x1,y1,width, height = person[0]['box']
    x1 , y1 = abs(x1) , abs(y1)
    x2= x1 + width
    y2 = y1 + height
    face = pixels[y1:y2,x1:x2]
    image = np.asarray(face)
    face_array = np.array(Image.fromarray(image.astype(np.uint8)).resize((160, 160)))

    return face_array

def load_faces(directory):
    faces = list()
    for filename in os.listdir(directory):
        path = os.path.join(directory,filename)
        face = extract_face(path)
        faces.append(face)
        
    return faces
    
def load_dataset(directory):
    X , y = list(), list()
    for subdir in os.listdir(directory):
        path = os.path.join(directory,subdir)
        faces = load_faces(path)
        labels = [subdir for _ in range(len(faces))]
        X.extend(faces)
        y.extend(labels)
        
    return asarray(X), asarray(y)




def get_embedding(model,face_pixels):
    #scaling pixel values
    face_pixels = face_pixels.astype('float32')
    #standardize pixel values
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = expand_dims(face_pixels, axis =0)
    yhat = model.predict(samples)
    return yhat[0]


def new_face(image):
    test_face_extract = extract_face(image)
    test_face_embedding = get_embedding(model,test_face_extract)
    #we can try to convert it into as array
    test_face_embedding = asarray(test_face_embedding)
    return test_face_embedding



def main():
    data_folder = load_dataset('ids')
    face_array = data_folder[0]
    face_name = data_folder[1].astype(object)
    savez_compressed('faces-dataset.npz', face_array,face_name)
    
    #Load dataset and save embeddings
    data = load('faces-dataset.npz',allow_pickle=True)
    model = load_model('facenet_mtcnn/facenet_keras.h5')
    print('Loaded model')
    faceArray,faceName = data['arr_0'], data['arr_1']
    face_embed = list()
    face_database = {}
    for face_pixels in faceArray:
        embedding = get_embedding(model,face_pixels)
        face_embed.append(embedding)
        face_embed = asarray(face_embed)
    savez_compressed('faces-embed.npz',face_embed,faceName)
    
    #get test image and its embedding
    test_face_embed = new_face(input("Enter the test image: "))
    
    #Load embedding
    load_embed = load('faces-embed.npz', allow_pickle= True)
    load_embed_faces = load_embed['arr_0']
    load_embed_names = load_embed['arr_1']
    
    #Normalize the embed vectors
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import Normalizer
    from sklearn.svm import SVC
    encoder = Normalizer(norm='l2')
    load_embed_faces1 = encoder.transform(load_embed_faces)
    #test_face_embed = encoder.transform(test_face_embed)
    # label encode targets
    name_encoder = LabelEncoder()
    name_encoder.fit(load_embed_names)
    load_embed_names1 = name_encoder.transform(load_embed_names)
    model = SVC (kernel='linear', probability= True, C=3)
    model.fit(load_embed_faces1,load_embed_names1)
    
    #to create one more array dimension 
    test_sample = expand_dims(test_face_embed,axis=0)
    yhat_class = model.predict(test_sample)
    yhat_prob = model.predict_proba(test_sample)
    #get name
    class_index = yhat_class[0]
    class_probability = yhat_prob[0,class_index] * 100
    predict_names = name_encoder.inverse_transform(yhat_class)
    print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
    
if __name__ == "__main__":
    main()


