import rclpy
import numpy as np
from vision.detector import ImageListener
import argparse
import json
import cv2
def main(args):
    rclpy.init()
    node = ImageListener(['fanta can','monster can', 'coffee can', 'lemon', 'redbull can',
                      'apple', 'starbucks can', 'orange', 'pepsi can', 'coca cola can'], person=True, yolo=args.on_board)
    #  'pepsi can'

    while True:
        node.rgb_cv = None
        while node.depth_cv is None or node.rgb_cv is None:
            print("waiting for image")
            rclpy.spin_once(node, timeout_sec=0.1)
        rclpy.spin_once(node, timeout_sec=1.0)

        objects_xyz, found_object_names = node.detect_object(save_indx=args.index)

        while node.human_cv is None:
            node.capture_human()

        human_boxes, human_names = node.detect_human(['person wearing green shirt', 'person wearing yellow shirt',
                    'person wearing black shirt', 'person wearing blue shirt', 'person wearing red shirt', 'person wearing brown shirt']
                                                    , save_indx=args.index)

        centers = [(box[0] + box[2])/2 for box in human_boxes]
        centers = np.array(centers)
        sorted_index = np.argsort(centers)
        human_data = []
        for i in range(len(centers)):
            indx = int(3 - sorted_index[i])
            human_data.append(indx)
        det_img1 = cv2.imread("./data/det/human_det_{}.jpg".format(args.index))
        det_img2 = cv2.imread("./data/det/det_{}.jpg".format(args.index))
        det_img1 = cv2.resize(det_img1, (640,480))
        det_img2 = cv2.resize(det_img2, (640,480))
        # visulize image
        cv2.imshow("human", det_img1)
        cv2.imshow("object", det_img2)
        k = cv2.waitKey(0)
        if k == ord('s'):
            object_names = [name.replace("a photo of a","") for name in found_object_names]
            human_names = [name.replace("a photo of a","") for name in human_names]

            det_res = {
                    "objects_xyz": objects_xyz, "object_name": object_names,
                    'person_name': human_names, 'person_data': human_data
                }

            with open("./data/det/demo_{}.json".format(args.index), "w") as f:
                json.dump(det_res, f, indent=4)
            print("save")
            break
        elif k == ord('q'):
            break

    node.destroy_node()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--on_board", type=int, default=0)
    args = parser.parse_args()
    main(args)