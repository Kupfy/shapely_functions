#!/usr/bin/python2.7
# coding=utf-8

from shapely.geometry import Polygon, LineString, Point
from shapely.affinity import rotate
import random
from math import radians, cos, sin, pi, sqrt
import numpy
from sys import float_info
from polygon import lines_of_polygon

MAX_ASPECT_RATIO = 15.0
MIN_ASPECT_RATIO = 1.0
MIN_HEIGHT = 0.0
MIN_WIDTH = 0.0
N_TRIES = 100
ANGLE_STEPS = 0.5
ANGLE_WIDTH = 5.0
ASPECT_RATIO_STEP = 0.25
WIDTH_STEP = 1e-4
AREA_SCALER = 10e10


def largest_rectangle(polygon, angles, tries=N_TRIES, spread=False):
    """
    Get the largest rectangle inside a polygon aligned to angles

    See Also:
        https://d3plus.org/blog/behind-the-scenes/2014/07/08/largest-rect/
    Args:
        polygon (Polygon): The polygon being checked
        angles (list(float)): The angles
        tries (int): The amount of random points to generate
        spread (bool): If you want to try a spread of angles generated around the `angles`

    Returns:
        Polygon: Largest rectangle
    """

    # Check that the polygon exists
    if polygon.area == 0:
        return None

    # Starting random points to use
    origins, width_step = points_inside_polygon(polygon, tries=tries)

    max_area = 0.0
    max_rect = None

    # generate the additional angles based on the spread
    if spread:
        temp_angles = []

        for angle in angles:
            temp_angles.extend(_angle_spread(angle, width=20.0, step=5.0))

        angles.extend(temp_angles)

    # add 90˚ angle for everything
    list({angles.extend([(a + 90) % 180 for a in angles[:]])})

    for angle in angles:
        # Convert the angle to radians
        angle_rad = radians(angle)

        for point in origins:
            modified_origins = []

            width = polygon_ray_cast(polygon, point, angle_rad)
            if width:
                modified_origins.append(width.centroid)

            height = polygon_ray_cast(polygon, point, (angle_rad + pi / 2))
            if height:
                modified_origins.append(height.centroid)

            for mod_point in modified_origins:
                mod_width = polygon_ray_cast(polygon, mod_point, angle_rad)

                if not mod_width:
                    continue

                min_dist_width = min(mod_point.distance(Point(mod_width.coords[0])),
                                     mod_point.distance(Point(mod_width.coords[1])))
                max_width = 2 * min_dist_width

                mod_height = polygon_ray_cast(polygon, mod_point, (angle_rad + pi / 2))

                if not mod_height:
                    continue

                min_dist_height = min(mod_point.distance(Point(mod_height.coords[0])),
                                      mod_point.distance(Point(mod_height.coords[1])))
                max_height = 2 * min_dist_height

                if max_width * max_height < max_area:
                    continue

                # TODO: Check this!! IN the c# code it had max_height * max_height which seems weird
                min_aspect_ratio = max(MIN_ASPECT_RATIO,
                                       numpy.float64(MIN_WIDTH) / max_height,
                                       numpy.float64(max_area) / (max_height ** 2))
                max_aspect_ratio = min(MAX_ASPECT_RATIO,
                                       numpy.float64(max_width) / MIN_HEIGHT,
                                       numpy.float64((max_width ** 2)) / max_area)

                # current_aspect_ratio = MIN_ASPECT_RATIO
                #
                # while current_aspect_ratio <= min_aspect_ratio + max_aspect_ratio:
                #     ratios.append(current_aspect_ratio)
                #     current_aspect_ratio += ASPECT_RATIO_STEP

                num_ratios = min(
                    max(0, int((max_aspect_ratio - min_aspect_ratio) / ASPECT_RATIO_STEP)), 10)
                ratios = numpy.linspace(min_aspect_ratio, max_aspect_ratio, num_ratios)

                # Go through all of the available ratios
                for boxRatio in ratios:
                    # Do a binary search to find the max width that works
                    left = max(MIN_WIDTH, sqrt(max_area * boxRatio))
                    right = min(max_width, max_height * boxRatio)

                    if right * max_height < max_area:
                        continue

                    # Try to get the largest rectangle that is completely inside of the polygon
                    while right - left >= width_step:
                        width = (left + right) / 2
                        height = width / boxRatio

                        rect_poly = Polygon([(mod_point.x - width / 2, mod_point.y - height / 2),
                                             (mod_point.x + width / 2, mod_point.y - height / 2),
                                             (mod_point.x + width / 2, mod_point.y + height / 2),
                                             (mod_point.x - width / 2, mod_point.y + height / 2)])

                        rect_poly = rotate(rect_poly,
                                           (pi/2 - angle_rad),
                                           mod_point,
                                           use_radians=True)

                        if polygon.contains(rect_poly):
                            max_area = width * height
                            max_rect = rect_poly
                            left = width
                        else:
                            right = width

    return max_rect


def points_inside_polygon(polygon, tries=N_TRIES):
    """
    Return random points inside a polygon

    TODO: potentially seed the random number generator to make it deterministic
    Args:
        polygon (Polygon): polygon to get points inside of
        tries (int): how many points to get

    Returns:
        list(Point): List of random points
    """

    bounds = polygon.bounds

    poly_width = bounds[2] - bounds[0]
    poly_height = bounds[3] - bounds[1]

    width_step = min(poly_width, poly_height) / 50

    # List of points inside polygon
    points = []

    # Get the geometric central point
    centre = polygon.centroid

    if polygon.contains(centre):
        points.append(centre)
    else:
        # If the centre, for some reason wasn't inside the polygon, add a cheap
        # representative point
        points.append(polygon.representative_point())

    # Create random points using the bounds of the polygon as the bounds
    # for the pseudo random number generator
    while len(points) < tries:
        rand_x = random.triangular(bounds[0], bounds[2])
        rand_y = random.triangular(bounds[1], bounds[3])

        rand_point = Point(rand_x, rand_y)
        if polygon.contains(rand_point) and rand_point not in points:
            points.append(rand_point)

    return points, width_step


def polygon_ray_cast(polygon, point, angle):
    """
    Cast a point along an angle onto a polygon.

    This results in a line that goes through the point and is bounded by the polygon

    See Also:
        https://d3plus.org/blog/behind-the-scenes/2014/07/08/largest-rect/
    Args:
        polygon (Polygon): Polygon to cast onto
        point (Point): Point to cast from
        angle (float): Angle in radians to cast along, starting at 0 on the x axis
    Returns:
        LineString: The line that was cast
    """

    eps = 1e-12

    adj_angle = pi / 2 - angle

    origin = Point(point.x + eps * cos(adj_angle), point.y + eps * sin(adj_angle))

    shifted_origin = Point(origin.x + cos(adj_angle), origin.y + sin(adj_angle))
    opp_shifted_origin = Point(origin.x - cos(adj_angle), origin.y - sin(adj_angle))

    idx = 0
    if abs(shifted_origin.x - origin.x) < eps:
        idx = 1
    i = -1

    poly_list = list(polygon.exterior.coords)
    n = len(poly_list)
    b = poly_list[n - 1]

    min_dist_left = float_info.max
    min_dist_right = float_info.max

    closest_point_left = None
    closest_point_right = None

    lines = lines_of_polygon(polygon)

    origin_line = LineString([shifted_origin, opp_shifted_origin])

    for line in lines:
        for l in line:
            if origin_line.intersects(l):
                intersect = origin_line.intersection(l)

                dist = origin.distance(l)

                if intersect.coords[0][idx] < origin.coords[0][idx]:
                    if dist < min_dist_left:
                        min_dist_left = dist
                        closest_point_left = intersect

                elif intersect.coords[0][idx] >= origin.coords[0][idx]:
                    if dist < min_dist_right:
                        min_dist_right = dist
                        closest_point_right = intersect

    if closest_point_left and closest_point_right:
        return LineString([closest_point_left, closest_point_right])
    else:
        return None


def _angle_spread(angle, width=ANGLE_WIDTH, step=ANGLE_STEPS):
    """
    Generate a list of angles within the ± ANGLE_WIDTH range around aAngle
    Args:
        angle (float): Angle to calculate around
        width (float): the spread of the angles to return
        step (float): the step between angles
    Returns:
        list(float): List of angles
    """

    min_angle = angle - width

    step_range = int(2 * width / step)

    return [(min_angle + x * step) % 180 for x in range(step_range + 1)]
