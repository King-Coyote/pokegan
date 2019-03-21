import pickle
from skimage import io
import numpy as np
import sys
import skimage

# use this to save datasets in array form. Pass as args all the images you want to save
# it will pickle them into the below file
f = open('pokesprites48.pkl', 'wb')
dumpArr = np.ndarray((len(sys.argv)-1, 48, 48, 3), dtype=np.float64)
i = 0
for file in sys.argv:
    if file[-4:] != '.png':
        continue
    im = io.imread(file)
    im.astype(np.float64)
    im = im/255.0
    im = im*2.0 - 1.0
    dumpArr[i,:,:,:] = skimage.transform.resize(im, (48,48))
    i += 1
pickle.dump(dumpArr, f)
f.close()