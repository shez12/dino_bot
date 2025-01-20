from  dino_vit_features import keypoint_utils
import mediapy as media
import numpy as np
import matplotlib.pyplot as plt
import cv2



def find_checkpoint(image_array1, image_array2,num_pairs = 5, load_size = 240):
    """
    Find the keypoint correspondences between two images.
    Args:
        image_array1: The first image.
        image_array2: The second image.
    Returns:
        patches_xy: The coordinates of the keypoints.
        desc1: The descriptors of the keypoints.
        descriptor_vectors: The descriptor vectors.
        num_patches: The number of patches.
    """
    patches_xy, desc1, descriptor_vectors, num_patches = keypoint_utils.extract_descriptors(image_array1, image_array2, num_pairs = num_pairs, load_size = load_size)
    return patches_xy, desc1, descriptor_vectors, num_patches


def extract_keypoints(image_array,descriptor_vectors, num_patches,load_size = 240):
    """
    Extract the keypoints from an image.
    Args:
        image_array: The image.
        descriptor_vectors: The descriptor vectors.
        num_patches: The number of patches.
        num_pairs: The number of pairs of keypoints to extract.
        load_size: The size to which the image is resized.
    Returns:
        cs_ys_list: The y-coordinates of the keypoints.
        cs_xs_list: The x-coordinates of the keypoints.
    """
    image_arrays = [image_array]
    map3, _ = keypoint_utils.extract_desc_maps(image_arrays,load_size=load_size)
    cs_ys_list, cs_xs_list = keypoint_utils.extract_descriptor_nn(descriptor_vectors, map3[0], num_patches, False)
    return cs_ys_list, cs_xs_list



if __name__ == "__main__":
    image1 = cv2.imread("image/camera1_rgb_0.jpg")
    image2 = cv2.imread("image/camera1_rgb_1.jpg")
    image3 = cv2.imread("image/camera1_rgb_2.jpg")
    print(image1.shape)

    patches_xy, desc1, descriptor_vectors, num_patches = find_checkpoint(image1, image2, num_pairs = 10, load_size = 240)

    image3 = cv2.resize(image2, (320,240))

    cs_ys_list3, cs_xs_list3 = extract_keypoints(image3, descriptor_vectors, num_patches,load_size=240)

    plt.figure()
    plt.imshow(image3)
    plt.scatter(cs_xs_list3, cs_ys_list3, c='r', s=5)
    plt.show()

