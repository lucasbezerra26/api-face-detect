from fastapi import FastAPI
from fastapi import File, UploadFile

from skimage.io import imread, imread_collection, imsave, imshow

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from skimage.transform import resize
import numpy as np

loaded_model = tf.keras.models.load_model("ResNet50.h5")
target_names = ["Bezerra", "Marcola do PCC"]

app = FastAPI()


@app.post("/")
async def face_detect(file: UploadFile = File(...)):
    try:
        contents = file.file.read()

        with open(file.filename, 'wb') as f:
            f.write(contents)

    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    imgs = imread_collection(file.filename)
    imgs = [np.array(resize(img, (384, 384))) for img in imgs]
    imgs = np.asarray(imgs)

    y_predicted = loaded_model.predict(preprocess_input(np.array(imgs)))
    max = y_predicted.argmax(axis=1)
    trust = y_predicted[0][max[0]]

    return {"face": target_names[max[0]], "trust": round(float(trust), 2)}