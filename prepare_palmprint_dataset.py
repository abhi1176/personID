
import cv2
import os
import numpy as np

from glob import glob
from PIL import Image
from skimage.transform import resize


palm_dir = os.path.join("datasets", "CASIA-PalmprintV1")
output_dir = os.path.join("datasets", "PalmCropped")


# Source: https://github.com/safwankdb/Effectual-Palm-RoI-Extraction/blob/master/ROI_extraction.ipynb
for img_file in glob(palm_dir+"/*/*"):
    print("[INFO] {}".format(img_file))
    try:
        image = Image.open(img_file)
        img_original = np.asarray(image)
        img_original = np.rot90(img_original, 3)  # Rotate 90 degree clockwise
        h, w = img_original.shape
        img = np.zeros((h + 160, w + 160), np.uint8)  # Pad the image by 80 pixes on 4 sides
        img[80:-80, 80:-80] = img_original
        # Apply GaussionBlur to remove noise
        blur = cv2.GaussianBlur(img, (5, 5), 0)

        # Apply Binary + OTSU thresholding to generate Black-White image
        # White pixels denote the palm and back pixels denote the background
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Section 2
        M = cv2.moments(th)
        h, w = img.shape
        # Get centroid of the white pixels
        x_c = M['m10'] // M['m00']
        y_c = M['m01'] // M['m00']

        # Apply Erosion to remove noise
        kernel = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]]).astype(np.uint8)
        erosion = cv2.erode(th, kernel, iterations=1)
        boundary = th - erosion

        cnt, _ = cv2.findContours(boundary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        img_c = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        areas = [cv2.contourArea(c) for c in cnt]
        max_index = np.argmax(areas)
        cnt = cnt[max_index]

        img_cnt = cv2.drawContours(img_c, [cnt], 0, (255, 0, 0), 2)

        cnt = cnt.reshape(-1, 2)
        left_id = np.argmin(cnt.sum(-1))
        cnt = np.concatenate([cnt[left_id:, :], cnt[:left_id, :]])

        # Section 3
        dist_c = np.sqrt(np.square(cnt-[x_c, y_c]).sum(-1))
        f = np.fft.rfft(dist_c)
        cutoff = 15
        f_new = np.concatenate([f[:cutoff], 0*f[cutoff:]])
        dist_c_1 = np.fft.irfft(f_new)

        # Section 4
        eta = np.square(np.abs(f_new)).sum()/np.square(np.abs(f)).sum()
        # print('Power Retained: {:.4f}{}'.format(eta*100, '%'))

        # Section 5
        derivative = np.diff(dist_c_1)
        sign_change = np.diff(np.sign(derivative))/2

        # Section 6
        minimas = cnt[np.where(sign_change>0)[0]]
        v1, v2 = minimas[-1], minimas[-3]

        theta = np.arctan2((v2-v1)[1], (v2-v1)[0])*180/np.pi
        print('The rotation of ROI is {:.02f}\u00b0'.format(theta))
        R = cv2.getRotationMatrix2D((int(v2[0]), int(v2[1])), theta, 1)

        img_r = cv2.warpAffine(img, R, (w, h))
        v1 = (R[:, :2] @ v1 + R[:, -1]).astype(np.int)
        v2 = (R[:, :2] @ v2 + R[:, -1]).astype(np.int)

        ux = v1[0]
        uy = v1[1] + (v2-v1)[0]//3
        lx = v2[0]
        ly = v2[1] + 4*(v2-v1)[0]//3

        roi = img_r[uy:ly, ux:lx]

        dirname = os.path.basename(os.path.dirname(img_file))
        filename = os.path.basename(img_file)
        output_file = os.path.join(output_dir, dirname, filename)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        cv2.imwrite(output_file, roi)
    except Exception as e:
        print("\n[ERROR] === Unable to find palm in: {}\n".format(img_file))
