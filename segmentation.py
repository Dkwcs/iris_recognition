
import signal
import numpy as np
import cv2


##### SEGMENTATION #####
def searchInnerBound(img):
    """
    Searching of the boundary (inner) of the iris
    """

    # integro-differential
    Y = img.shape[0] # height
    X = img.shape[1] # width
    sect = X/4
    minrad = 10
    maxrad = sect*0.8
    jump = 4 		# Precision of the search
    # print("Y={}, X={}, sect={}, minrad={}, maxrad={}, jump={}".format(Y, X, sect, minrad, maxrad, jump))

    # Hough Space
    sz = np.array([np.floor((Y-2*sect)/jump),
                    np.floor((X-2*sect)/jump),
                    np.floor((maxrad-minrad)/jump)]).astype(int)
    # print("sz={}".format(sz))

    #circular integration
    integrationprecision = 1
    angs = np.arange(0, 2*np.pi, integrationprecision)
    x, y, r = np.meshgrid(np.arange(sz[1]),
                          np.arange(sz[0]),
                          np.arange(sz[2]))
    y = sect + y*jump
    x = sect + x*jump
    r = minrad + r*jump
    # print("x.shape={}, y.shape={}, x={}, y={}, r={}".format(x.shape, y.shape, x, y, r))
    hs = ContourIntegralCircular(img, y, x, r, angs)

    # Hough Space Partial Derivative 
    hspdr = hs - hs[:, :, np.insert(np.arange(hs.shape[2]-1), 0, 0)]

    # blurring the image
    sm = 3
    hspdrs = signal.fftconvolve(hspdr, np.ones([sm,sm,sm]), mode="same")

    indmax = np.argmax(hspdrs.ravel())
    y,x,r = np.unravel_index(indmax, hspdrs.shape)

    inner_y = sect + y*jump
    inner_x = sect + x*jump
    inner_r = minrad + (r-1)*jump

    # Integro-Differential 
    integrationprecision = 0.1
    angs = np.arange(0, 2*np.pi, integrationprecision)
    x, y, r = np.meshgrid(np.arange(jump*2),
                          np.arange(jump*2),
                          np.arange(jump*2))
    y = inner_y - jump + y
    x = inner_x - jump + x
    r = inner_r - jump + r
    hs = ContourIntegralCircular(img, y, x, r, angs)

    # Hough Space Partial Derivative 
    hspdr = hs - hs[:, :, np.insert(np.arange(hs.shape[2]-1), 0, 0)]

    # blurring the image
    sm = 3 	
    hspdrs = signal.fftconvolve(hspdr, np.ones([sm,sm,sm]), mode="same")
    indmax = np.argmax(hspdrs.ravel())
    y,x,r = np.unravel_index(indmax, hspdrs.shape)

    inner_y = inner_y - jump + y
    inner_x = inner_x - jump + x
    inner_r = inner_r - jump + r - 1

    return inner_y, inner_x, inner_r

def searchOuterBound(img, inner_y, inner_x, inner_r):
    """
    Searching fo the boundary (outer) of the iris 
    """
    maxdispl = np.round(inner_r*0.15).astype(int)

    minrad = np.round(inner_r/0.8).astype(int)
    maxrad = np.round(inner_r/0.3).astype(int)

    # Integration region and avoiding eyelids
    intreg = np.array([[2/6, 4/6], [8/6, 10/6]]) * np.pi

    #circular integration
    integrationprecision = 0.05
    angs = np.concatenate([np.arange(intreg[0,0], intreg[0,1], integrationprecision),
                            np.arange(intreg[1,0], intreg[1,1], integrationprecision)],
                            axis=0)
    x, y, r = np.meshgrid(np.arange(2*maxdispl),
                          np.arange(2*maxdispl),
                          np.arange(maxrad-minrad))
    y = inner_y - maxdispl + y
    x = inner_x - maxdispl + x
    r = minrad + r
    hs = ContourIntegralCircular(img, y, x, r, angs)

    # Hough Space Partial Derivative
    hspdr = hs - hs[:, :, np.insert(np.arange(hs.shape[2]-1), 0, 0)]

    # blurring
    sm = 7 	# Size of the blurring mask
    hspdrs = signal.fftconvolve(hspdr, np.ones([sm, sm, sm]), mode="same")

    indmax = np.argmax(hspdrs.ravel())
    y, x ,r = np.unravel_index(indmax, hspdrs.shape)

    outer_y = inner_y - maxdispl + y + 1
    outer_x = inner_x - maxdispl + x + 1
    outer_r = minrad + r - 1

    return outer_y, outer_x, outer_r

def ContourIntegralCircular(imagen, y_0, x_0, r, angs):
    """
       Contour/circular integral using discrete rieman
    """
    y = np.zeros([len(angs), r.shape[0], r.shape[1], r.shape[2]], dtype=int)
    x = np.zeros([len(angs), r.shape[0], r.shape[1], r.shape[2]], dtype=int)
    for i in range(len(angs)):
        ang = angs[i]
        y[i, :, :, :] = np.round(y_0 - np.cos(ang) * r).astype(int)
        x[i, :, :, :] = np.round(x_0 + np.sin(ang) * r).astype(int)
    
    # adapt x and y
    ind = np.where(y < 0)
    y[ind] = 0
    ind = np.where(y >= imagen.shape[0])
    y[ind] = imagen.shape[0] - 1
    ind = np.where(x < 0)
    x[ind] = 0
    ind = np.where(x >= imagen.shape[1])
    x[ind] = imagen.shape[1] - 1

    hs = imagen[y, x]
    hs = np.sum(hs, axis=0)
    return hs.astype(float)

def segment(eyeim):
    """
        Segment the iris from the image
    """
    segmented_img = eyeim
    
    # Using daugman intefro-differential to the the iris
    # search the inner and outer bounds
    rowp, colp, rp = searchInnerBound(eyeim)
    row, col, r = searchOuterBound(eyeim, rowp, colp, rp)

    # pupil and iris boundaries
    rowp = np.round(rowp).astype(int)
    colp = np.round(colp).astype(int)
    rp = np.round(rp).astype(int)
    row = np.round(row).astype(int)
    col = np.round(col).astype(int)
    r = np.round(r).astype(int)
    cirpupil = [rowp, colp, rp]
    ciriris = [row, col, r]

    # Crop iris by circles
    mask1 = np.zeros_like(segmented_img)
    mask1 = cv2.circle(mask1, (colp, rowp), rp, (255, 255, 255), -1)
    mask2 = np.zeros_like(segmented_img)
    mask2 = cv2.circle(mask2, (col, row), r, (255, 255, 255), -1)
    mask = cv2.subtract(mask2, mask1)
    segmented_img = cv2.bitwise_and(segmented_img, segmented_img, mask=mask)

    # cut iris image
    imsz = segmented_img.shape
    irl = np.round(row - r).astype(int)
    iru = np.round(row + r).astype(int)
    icl = np.round(col - r).astype(int)
    icu = np.round(col + r).astype(int)
    if irl < 0:
        irl = 0
    if icl < 0:
        icl = 0
    if iru >= imsz[0]:
        iru = imsz[0] - 1
    if icu >= imsz[1]:
        icu = imsz[1] - 1
    segmented_img = segmented_img[irl: iru + 1, icl: icu + 1]

    return segmented_img, cirpupil, ciriris