import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import load_model
saved_model = load_model("model/model_DensNet.h5")
status = True


def check(input_img):
    print(" your image is : " + input_img)
    print(input_img)

    img = image.load_img("images/" + input_img, target_size=(100, 100))
    img = np.asarray(img)
    print(img)

    img = np.expand_dims(img, axis=0)

    print(img)
    output = saved_model.predict_classes(img)

    print(output)
    if output == 1:
        status = True
    else:
        status = False

    print(status)
    return status
