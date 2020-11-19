import argparse
import cv2
import numpy as np


def get_area(source,template,s_g,t_g,h,w):
    """
    :param source: Color source image (StarMap.png)
    :param template: Color template image (Small_area.png/Small_area_rotated.png)
    :param s_g: Grayscale source image
    :param t_g: Grayscale template image
    :param h: template_g.shape[0]
    :param w: temlate_g.shape[1]
    :return: None

    This function is used for determine the 4 point positions of template image in the Star Map image.
    If image is not rotated Template Matching will determine exact locations.
    If image is rotated Feature Matching and Homography case will determine approximate locations

    """

    result = cv2.matchTemplate(s_g, t_g, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    top_left = max_loc
    bottom_right = (top_left[0] + h, top_left[1] + w)
    top_right = (top_left[0], bottom_right[1])
    bottom_left = (bottom_right[0], top_left[1])


    template_mean = np.mean(template)   # Mean of template image
    template_std = np.std(template)     # Standart Deviation of template image

    roi = source[top_left[1]:top_left[1] + w, top_left[0]:top_left[0] + h] # roi refers to proposed locations in source image
    roi_mean = np.mean(roi)   # Mean of roi in source image
    roi_std = np.std(roi)     # Standart Deviation of roi in source image

    if (template_mean == roi_mean and template_std == roi_std): # If mean and std of roi equal to mean and std of template image , these locations are correct
        print("Coordinates: {} , {}, {}, {}".format(top_left, top_right, bottom_right,bottom_left))
    else:
        # If code goes here, it means image is rotated
        del roi,roi_std,roi_mean

        orb = cv2.ORB_create(nfeatures=10000) # Initiate ORB object with 100000 features

        #find the keypoints and descriptors
        keypoints_s, descriptors_s = orb.detectAndCompute(s_g,None)
        keypoints_t, descriptors_t = orb.detectAndCompute(t_g,None)


        flann_index_lsh = 6
        index_params = dict(algorithm = flann_index_lsh)
        search_params = dict()
        # Set Flann Based Matcher
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        # Get matches
        flann_matches = flann.knnMatch(descriptors_t,descriptors_s,k=2)


        good_matches = list()
        for x,y in flann_matches:
            if(x.distance < 0.7 * y.distance):
                good_matches.append(x)

        src_pts = np.float32([keypoints_t[x.queryIdx].pt for x in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([keypoints_s[x.trainIdx].pt for x in good_matches]).reshape(-1,1,2)

        M, mask = cv2.findHomography(srcPoints=src_pts,dstPoints=dst_pts,method=cv2.RANSAC)

        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M).astype(np.int)
        print("Coordinates: {} , {}, {}, {}".format(dst[0],dst[1],dst[2],dst[3]))


if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument("-s", "--source", required=True,type=str, help="Path to source image // Must be StarMap.png file")
    parse.add_argument("-t", "--template", required=True,type=str, help="Path to template image")

    args = vars(parse.parse_args())

    source = cv2.imread(args["source"],cv2.IMREAD_ANYCOLOR)
    template = cv2.imread(args["template"], cv2.IMREAD_ANYCOLOR)

    source_g = cv2.imread(args["source"],0)
    template_g = cv2.imread(args["template"],0)

    h, w = template_g.shape
    get_area(source,template,source_g,template_g,w,h)