import numpy as np

def rle2mask(rle, height=1024, width=1024):
	mask= np.zeros(width* height)
	array = np.asarray([int(x) for x in rle.split()])
	starts = array[0::2]
	lengths = array[1::2]

	current_position = 0
	for index, start in enumerate(starts):
	    current_position += start
	    mask[current_position:current_position+lengths[index]] = 255
	    current_position += lengths[index]

	return mask.reshape(width, height)

def run_length_encode(component):
    component = component.T.flatten()
    start = np.where(component[1:] > component[:-1])[0]+1
    end = np.where(component[:-1] > component[1:])[0]+1
    length = end-start
    rle = []
    for i in range(len(length)):
        if i == 0:
            rle.extend([start[0], length[0]])
        else:
            rle.extend([start[i]-end[i-1], length[i]])
    rle = ' '.join([str(r) for r in rle])
    return rle

def predict(X, threshold):
    X_p = np.copy(X)
    preds = (X_p > threshold).astype('uint8')
    return preds