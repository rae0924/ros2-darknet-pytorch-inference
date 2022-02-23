import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import ros2_numpy as rnp
import cv2

from ament_index_python.packages import get_package_share_directory
import os
import time

from .tool.utils import *
from .tool.torch_utils import *
from .tool.darknet2pytorch import Darknet

class DarknetInferenceNode(Node):

    def __init__(self, config_file, weights_file, names_file):
        super().__init__('darknet_inference_node')

        self.config_file = config_file
        self.weights_file = weights_file
        self.names_file = names_file

        self.model = Darknet(self.config_file)
        self.model.load_weights(self.weights_file)
        self.model.cuda()
        
        self.class_names = load_class_names(names_file)

        # change subscription if needed
        self.color_image_sub = self.create_subscription(
            Image, 
            '/zed2/zed_node/rgb/image_rect_color',
            self.rgb_image_callback,
            10
        )

        
    def rgb_image_callback(self, msg: Image):
        image = rnp.numpify(msg)
        image = image[..., :3]
        sized = cv2.resize(image, (self.model.width, self.model.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        start = time.time()
        boxes = do_detect(self.model, sized, 0.4, 0.4)
        finish = time.time()

        time_elapsed = finish - start
        self.get_logger().info('Finished inference: %f ms, %f fps' % (time_elapsed, 1/time_elapsed))

        plot_boxes_cv2(image, boxes[0], savename='predictions.jpg', class_names=self.class_names)


def main(args=None):
    rclpy.init(args=args)

    share_dir = os.path.dirname(get_package_share_directory('darknet_inference'))
    names_file = os.path.join(share_dir, 'names/coco.names')
    config_file = os.path.join(share_dir, 'cfg/yolov4.cfg')
    weights_file = os.path.join(share_dir, 'weights/yolov4.weights')

    inference_node = DarknetInferenceNode(config_file, weights_file, names_file)

    rclpy.spin(inference_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    inference_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()