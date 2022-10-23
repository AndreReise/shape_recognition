import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import numpy as np
import math
import matplotlib

matplotlib.use('TkAgg')

MAX_INT = sys.maxsize
MIN_INT = -1 * sys.maxsize

state = 0


def show(sample_name):
    img = plt.imread(sample_name)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # weighted method to convert RGB(a) to Grayscale
    # also use constant thresholding - 0.5
    gray_scale = (0.33 * r + 0.33 * g + 0.33 * b) > 0.5

    small_patches = []

    window_size = 15

    x_sq = 0

    nodes_count = int(550 / window_size)
    visited_nodes = np.full((nodes_count, nodes_count), 0)
    nodes = np.full((nodes_count, nodes_count), 0)

    # function to iterate  through all pixels in the capture window 
    def contains(x_start, x_end, y_start, y_end):

        # extra condition to not access out of bound index
        # don't want to review algo implementation...
        if x_end > 550:
            x_end = 550
        if y_end > 550:
            y_end = 550

        for x in range(x_start, x_end):
            for y in range(y_start, y_end):
                if not gray_scale[x, y]:
                    return True
        return False

    while (x_sq + 1) * window_size <= 550:
        y_sq = 0
        while (y_sq + 1) * window_size <= 550:
            if contains(x_sq * window_size, (x_sq + 1) * window_size, y_sq * window_size, (y_sq + 1) * window_size):
                nodes[x_sq, y_sq] = True
                square = patches.Rectangle(
                    (y_sq * window_size, x_sq * window_size), window_size, window_size, alpha=0.5)
                small_patches.append(square)

            y_sq = y_sq + 1
        x_sq = x_sq + 1

    fig, ax = plt.subplots()

    shapes_colors = [i[0] for i in colors.TABLEAU_COLORS.items()]
    shape_index = 0
    shapes = []

    class Shape:
        def __init__(self, x_min, x_max, y_min, y_max, w_size, name):
            self.x_min = x_min * w_size
            self.x_max = x_max * w_size
            self.y_min = y_min * w_size
            self.y_max = y_max * w_size
            self.w_size = w_size
            self.x_center = int(((x_min + 1) * w_size + x_max * w_size) / 2)
            self.y_center = int(((y_min + 1) * w_size + y_max * w_size) / 2)
            self.name = name

        def get_height(self):
            return self.x_max - self.x_min + self.w_size

        def get_width(self):
            return self.y_max - self.y_min + self.w_size

    def visit_node(x, y, x_min, x_max, y_min, y_max):

        if x < 0 or x >= nodes_count or y < 0 or y >= nodes_count:
            return MAX_INT, MIN_INT, MAX_INT, MIN_INT

        if visited_nodes[x, y] or not nodes[x, y]:
            return MAX_INT, MIN_INT, MAX_INT, MIN_INT

        visited_nodes[x, y] = True

        (x_min1, x_max1, y_min1, y_max1) = visit_node(x + 1, y, x_min, x + 1, y_min, y_max)
        (x_min2, x_max2, y_min2, y_max2) = visit_node(x, y + 1, x_min, x_max, y_min, y + 1)
        (x_min3, x_max3, y_min3, y_max3) = visit_node(x - 1, y, x - 1, x_max, y_min, y_max)
        (x_min4, x_max4, y_min4, y_max4) = visit_node(x, y - 1, x_min, x_max, y - 1, y_max)

        return (
            min(x_min, x_min1, x_min2, x_min3, x_min4),
            max(x_max, x_max1, x_max2, x_max3, x_max4),
            min(y_min, y_min1, y_min2, y_min3, y_min4),
            max(y_max, y_max1, y_max2, y_max3, y_max4))

    for x in range(0, nodes_count):
        for y in range(0, nodes_count):
            # skip if we already detect this object
            if nodes[x, y] and not visited_nodes[x, y]:
                (x_min, x_max, y_min, y_max) = visit_node(x, y, x, x, y, y)

                visited_shape = Shape(x_min, x_max, y_min, y_max, window_size, shapes_colors[shape_index])

                # rectangle patch
                square_patch = patches.Rectangle(
                    (visited_shape.y_min, visited_shape.x_min), visited_shape.get_width(),
                    visited_shape.get_height(), color=shapes_colors[shape_index], alpha=0.7)
                ax.add_patch(square_patch)

                # dot
                plt.plot(visited_shape.y_center, visited_shape.x_center, 'go')

                visited_nodes[x, y] = True
                shapes.append(visited_shape)
                shape_index += 1

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    for patch in small_patches:
        ax.add_patch(patch)

    plt.imshow(img)
    plt.show()
    return shapes


def calc_and_print_distance(shape1, shape2):
    distance = math.sqrt(math.pow(shape1.x_center - shape2.x_center, 2)
                         + math.pow(shape1.y_center - shape2.y_center, 2))

    print("Shape {} | Shape {} | Distance {}".format(shape1.name, shape2.name, distance))


image_count = len(os.listdir("./img"))
index = 0

# for i in range(image_count):
#     path = "img/sample{}.png".format(i)
#     shapes = show(path)
#     distances = [calc_and_print_distance(shapes[p1], shapes[p2]) for p1 in range(len(shapes)) for p2 in
#                  range(p1 + 1, len(shapes))]
#     print("\n\n")

while True:
    shapes = show("img/sample{}.png".format(index))
    distances = [calc_and_print_distance(shapes[p1], shapes[p2])
                 for p1 in range(len(shapes))
                 for p2 in range(p1 + 1, len(shapes))]

    key = input()

    if key == 'a' and index > 0:
        index = index - 1
    elif key == 'd' and index < (image_count - 1):
        index = index + 1
    elif key == 'e':
        break
