# Pneumothorax-Segmentation

TODO:

1. CRF
2. TTA
3. CYCLIC LR
4. THRESHOLD using IOU score
5. Check quality of image downsampling (DOG - difference of Gaussian), Information loss (FFT)
6. Lossless image compression


#### Postprocessing

def dice_overall(preds, targs):
    n = preds.shape[0]
#     preds = preds.view(n, -1)
#     targs = targs.view(n, -1)
    intersect = (preds * targs).sum()
    union = (preds+targs).sum()
#     u0 = union==0
#     intersect[u0] = 1
#     union[u0] = 2
    return (2. * intersect / union)
    
dices = []
thrs = np.arange(0.01, 1, 0.01)
for i in tqdm_notebook(thrs):
    preds_m = (preds_valid>i)
    dices.append(dice_overall(preds_m, Y_VALID).mean())
dices = np.array(dices)


https://www.kaggle.com/meaninglesslives/unet-xception-keras-for-pneumothorax-segmentation
