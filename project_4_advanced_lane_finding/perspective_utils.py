import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from calibration_utils import calibrate_camera, undistort
from binarization_utils import binarize


class Warp:

    def __init__(self):
        self.h = 0
        self.v = 0
        pass

    def calc_matrices(self, img):
        h, w = img.shape[:2]
        if self.h != h or self.w != w:
            self.h, self.w = img.shape[:2]

            self.src = np.float32([
                [w, h - (h * 0.055)],         # br
                [0, h - (h * 0.055)],         # bl
                [w * 0.383, h * 0.638],       # tl
                [w - (w * 0.383), h * 0.638]  # tr
            ])

            self.dst = np.float32([
                [w - (w * 0.156), h],  # br
                [w * 0.156, h],        # bl
                [0, 0],                # tl
                [w, 0]                 # tr
            ])

            self.matrix = cv2.getPerspectiveTransform(self.src, self.dst)
            self.inverse_matrix = cv2.getPerspectiveTransform(self.dst, self.src)

    def perspective(self, img, birdeye=True, verbose=False):
        """
        Apply perspective transform to input frame to get the bird's eye or perspective view.
        :param img: input color frame
        :param birdeye: if True, get bird's eye transformation, else (inverse) perspective
        :param verbose: if True, show the transformation result
        :return: warped image
        """
        h, w = img.shape[:2]
        if self.h != h or self.w != w:
            self.calc_matrices(img)

        if birdeye:
            matrix = self.matrix
        else:
            matrix = self.inverse_matrix

        warped = cv2.warpPerspective(img, matrix, (w, h), flags=cv2.INTER_LINEAR)

        if verbose:
            f, axarray = plt.subplots(1, 2)
            f.set_facecolor('white')
            axarray[0].set_title('Before perspective transform')
            axarray[0].imshow(img, cmap='gray')
            for point in self.src:
                axarray[0].plot(*point, '.')
            axarray[1].set_title('After perspective transform')
            axarray[1].imshow(warped, cmap='gray')
            for point in self.dst:
                axarray[1].plot(*point, '.')
            for axis in axarray:
                axis.set_axis_off()
            plt.show()

        return warped


if __name__ == '__main__':

    ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='camera_cal')

    # show result on test images
    for test_img in glob.glob('test_images/*.jpg'):

        img = cv2.imread(test_img)

        img_undistorted = undistort(img, mtx, dist, verbose=False)

        img_binary = binarize(img_undistorted, verbose=False)

        warp = Warp()
        img_birdeye = warp.perspective(cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2RGB), verbose=True)
