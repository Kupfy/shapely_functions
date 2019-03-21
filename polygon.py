#!/usr/bin/python2.7
# coding=utf-8
"""
Author: Adam Kupferman
Useful functions to use with polygons with the shapely library

Note:
    All functions assume that the if the points are using latitude and longitude 
    that they are in the form Point(lng, lat) since this is the equivalent of
    Point(x, y)
"""


from math import sqrt, radians, sin, cos, degrees, atan2, acos
from shapely.geometry import LinearRing, LineString, Polygon, Point
from shapely.validation import explain_validity
import itertools
from shapely.ops import transform
from functools import partial
import pyproj
from decimal import Decimal

# the radius of the earth in m
R = 6378136.98


def encode_polygon(polygon, accuracy=6):
    """
    Encode a polygon to a single string using the Bing maps point compression

    Notes:
        https://developers.google.com/maps/documentation/utilities/polylinealgorithm
    Args:
        polygon (Polygon): unencoded polygon
        accuracy(int): number of decimal places to encode

    Returns:
        str: encoded string
    """

    points = list(polygon.exterior.coords)

    multiplier = float("1e%s" % accuracy)

    encoded = ''

    # get the decimal precision of the point, so that we don't get unwanted
    # floating point error
    precision = lambda x: abs(Decimal(str(x)).as_tuple().exponent)

    for i, point in enumerate(points[:-1]):
        if i == 0:
            point_diff = point
        else:
            point_diff = (point[0] - points[i - 1][0], point[1] - points[i - 1][1])

        # Get the precision of the point. Only up to 6 decimal places will be used as
        # any more is both unnecessary and untrustworthy

        dp0 = min(precision(point_diff[0]), accuracy)
        dp1 = min(precision(point_diff[1]), accuracy)

        encoded += ''.join((encode_point(round(point_diff[0], dp0), multiplier),
                            encode_point(round(point_diff[1], dp1), multiplier)))

    return encoded


def encode_point(point, multiplier=1e6):
    """
    Encode a single float to a string using the Bing maps point compression
    Args:
        point (float): unencoded point
        multiplier (float): multiplier to get rid of decimal places

    Returns:
        str: encoded point
    """

    rounded = int(point * multiplier)
    # left shift by 1
    rounded <<= 1

    # If the number is negative, invert it (this means that the last bit will be 1
    # which will let us know that it is negative when decoding
    if rounded < 0:
        rounded = ~rounded

    # break it up into bit groups
    split = []
    while rounded > 0:
        # take the 5 right bits
        chunk = rounded & 0x1F
        split.append(chunk)

        # remove those 5 bits
        rounded >>= 5

    if len(split) == 0:
        split = [0]

    # OR chunks by 0x20 if they are followed by another chunk,
    # convert to decimal, add 63 and convert to ascii
    encoded = ''.join([chr((s | 0x20) + 63) for s in split[:-1]])
    encoded += chr(split[-1] + 63)

    return encoded


def decode_polygon(encoded, accuracy=6):
    """
    Decode a string into a Polygon using the Bing maps point compression

    Args:
        encoded (str): the encoded Polygon
        accuracy(int): number of decimal places to decode for

    Returns:
        Polygon: decoded polygon
    """

    # the coords of the polygon
    poly_coords = []
    # the bit chunks used to make a point
    point_chunks = []
    # if it is the lat or lng value
    lng = True
    point = []

    multiplier = float("1e%s" % accuracy)

    for char in encoded:
        # Decode ascii to integer
        dec_char = ord(char) - 63

        # characters that weren't bitwise ORed by 0x20 are the final chunk
        # in the number's encoding
        if dec_char & 0x20:
            dec_char &= 0x1F
            point_chunks.append(dec_char)
        else:
            point_chunks.append(dec_char)
            decoded = decode_point(point_chunks, multiplier)
            point_chunks = []

            if lng:
                point = [round(decoded, accuracy)]
            else:
                point.append(round(decoded, accuracy))

                # Add the coordinates to the list of coordinates for the polygon
                # Points in the encoded string are only a difference from the previous
                # point, so add the decoded point to the previous one
                if len(poly_coords) > 0:
                    poly_coords.append((round(poly_coords[-1][0] + point[0], accuracy),
                                       round(poly_coords[-1][1] + point[1], accuracy)))
                else:
                    poly_coords.append(tuple(point))

            lng = not lng

    return Polygon(poly_coords)


def decode_point(chunks, multiplier=1e6):
    """
    Decode the chunks of a point into a single float using the Bing maps point compression

    Args:
        chunks (list(int)): chunks of the binary point in reverse order
        multiplier(float): multiplier to get rid of decimal places based on how many were encoded

    Returns:
        float: the point
    """

    number = 0

    for i, chunk in enumerate(chunks):
        # shift the number the correct number of 0s to make it a single integer
        number |= chunk << (i * 5)

    # if the number was negative it will have a 1 as the final binary
    # value
    if number & 0x1:
        number = ~number

    # Right shift by 1 to get rid of the last 0 bit
    number >>= 1

    # convert to a float
    number /= multiplier

    return number


def remove_collinear_points(polygon, threshold=1.0):
    """
    Removes points in a polygon that lie on the straight line created by its
    adjacent points.

    aThreshold defaults to 1, less than 1 will be more precise (remove less points) and greater
    than 1 will be less precise (remove more points)

    Args:
        polygon (shapely.geometry.Polygon): the shape to be simplified
        threshold (float): Accuracy threshold. Default: 1

    Returns:
        The simplified polygon
    """
    if threshold < 0.0:
        print("Threshold must be greater than 0")
        return None

    points = list(polygon.exterior.coords)

    # Threshold of how far a point can be from a linestring in order to define it as collinear
    threshold = threshold * 10 ** (-6)

    index = 0

    while index != len(points) - 1:
        linestring = LineString([points[index - 1], points[index + 1]])

        if 0 < linestring.distance(Point(points[index])) < threshold:
            del points[index]
        else:
            index += 1

    return Polygon(points)


def join_outlines(outlines):
    """
    Join a list of outlines to create a single polygon. Error if there is a polygon that didn't
    touch the others
    Args:
        outlines (list(Polygon)): list of polygons to join together

    Returns:
        joined polygon
    """

    # If they didn't all join upon creation, try to join them together by looping through the
    # list until we only have 1 shape left. The mega shape
    count = 0

    while len(outlines) > 1:
        # count to hit to make sure we don't get into an endless loop if a bad parcel was put in
        exit_count = len(outlines) / 2

        for (shape1, shape2) in itertools.combinations(outlines, 2):
            if shape1.touches(shape2):
                new_union = shape1.union(shape2)
                outlines.append(new_union)

                outlines.remove(shape1)
                outlines.remove(shape2)

                count -= 1
                break

        count += 1

        if count > exit_count:
            print("Exiting outline join. There is an outline(s) that doesn't touch others")
            return None

    return outlines[0]


def join_outlines_multi(outlines):
    """
    Join a list of polygons to the lowest number of objects possible
    Args:
        outlines (list(Polygon)): polygons to join
    Returns:
        GeometryObject:
    """

    # bool to make sure we have have polygons to join
    touch = True

    while touch:
        for (shape1, shape2) in itertools.combinations(outlines, 2):
            if shape1.intersects(shape2):
                new_union = shape1.union(shape2)
                outlines.append(new_union)

                outlines.remove(shape1)
                outlines.remove(shape2)

                break
        else:
            touch = False

    return outlines


def trans_4326_to_AEA(polygon):
    """
    Transform a polygon from EPSG:4326 to AEA.

    The AEA projection requires the proj, lat1/2 parameters which is why its not as easy
    to make this function more generalised

    See:
        https://epsg.io has info on different

    Args:
        polygon (Polygon): polygon to be transformed

    Returns:
        (Polygon): transformed shape
    """

    transformed_shape = transform(
        partial(
            pyproj.transform,
            pyproj.Proj(init="epsg:4326"),
            pyproj.Proj(
                proj="aea",
                lat1=polygon.bounds[1],
                lat2=polygon.bounds[3],
            )
        ), polygon
    )

    return transformed_shape


def area_of_polygon(polygon):
    """
    Calculates the area of a polygon that uses lat lng as points in m^2

    Note:
        This uses the Albers equal-area projection.


    Args:
        polygon (Polygon):

    Returns:
        Area in m²
    """

    if len(polygon.bounds) < 4:
        return 0

    return trans_4326_to_AEA(polygon).area


def lines_of_polygon(polygon):
    """
    Split a polygon into a list of lines. The returning value will contain nested
    lists where the first list is the shell's lines and any following lists are
    the holes of the polygon

    Args:
        polygon (Polygon): Polygon to be split

    Returns:
        list(list(LineString))):
    """

    bounds = polygon.boundary

    multiLines = []

    if bounds.geom_type == "MultiLineString":
        for linestring in bounds:
            lines = lines_of_line_string(linestring)
            multiLines.append(lines)
    elif bounds.geom_type == "LineString":
        lines = lines_of_line_string(bounds)
        multiLines.append(lines)
    else:
        return None

    return multiLines


def closest_line_string_in_poly(polygon, line, min_distance=None, closest_line=None):
    """
    Find the linestring in a polygon outline that is closest to a line. Can also be
    used with a specific distance and line when comparing against a number of aLine
    in a loop
    Args:
        polygon (Polygon): Polygon to check against line
        line (LineString): Line to check against
        min_distance (float): Specific minimum distance to beat
        closest_line (LineString): LineString that is aMinDistance from aLine
    Returns:
        LineString, float: The closest linestring in aPolygon to aLine
    """
    # a list of lines that make up the outline of a polygon
    lines = lines_of_polygon(polygon)

    # distance between polygon and line according to shapely's algorithm
    distance = polygon.distance(line)

    # updating min distance to aLine
    min_distance = min_distance if min_distance else lines[0][0].distance(line)
    closest_line = closest_line if closest_line else lines[0][0]

    for line in lines[0]:
        centroid = line.centroid
        tempDist = centroid.distance(line)

        if tempDist < min_distance:
            min_distance = tempDist
            closest_line = line

        # If the minDistance has matched or gone bellow what shapely believes the
        # min distance is, its safe to say the closest linestring has been found
        if min_distance <= distance:
            break

    return closest_line, min_distance


def compass_bearing(linestring):
    """
    Calculate the bearing of line

    Note:
        The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))

    Args:
        linestring (LineString):

    Returns:
        float: the compass bearing
    """

    # x is longitude and y is latitude
    x, y = linestring.xy

    lat1 = radians(y[0])
    lat2 = radians(y[1])

    diff_lng = radians(x[1] - x[0])

    a = sin(diff_lng) * cos(lat2)
    b = cos(lat1) * sin(lat2) - (sin(lat1) * cos(lat2) * cos(diff_lng))

    initial_bearing = atan2(a, b)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = degrees(initial_bearing)
    bearing = (initial_bearing + 360) % 360

    return bearing


def haversine(point1, point2):
    """
    Calculate the distance between 2 points over the surface of the earth

    Notes:
        a = sin²(Δφ/2) + cos φ₁ ⋅ cos φ₂ ⋅ sin²(Δλ/2)
        c = 2 ⋅ atan2( √a, √(1−a) )
        d = R ⋅ c

        φ is latitude, λ is longitude, R is earth’s radius (mean radius = 6,371km);

    Args:
        point1 (Point): a point in format (lng, lat)
        point2 (Point): a point in format (lng, lat)

    Returns:
        float: distance in m
    """

    phi1 = radians(point1.y)
    phi2 = radians(point2.y)

    delta_phi = radians(point2.y - point1.y)
    delta_lambda = radians(point2.x - point1.x)

    a = sin(delta_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    d = R * c

    return d


def largest_rectangle_angle(polygon, linestrings):
    """
    Find the angle to align the largest rectangle in a polygon. Use a list of
    linestrings (road) to find the closest side of the polygon, and then take
    the perpendicular angle to that side

    Note:
        This will not find the absolute largest rectangle, simply find the closest
        when restricted to an angle. This is much faster than an absolute, and good
        enough for most of our needs
    Args:
        polygon (Polygon): Polygon to find angle
        linestrings (list(LineString)): list of linestrings to compare against

    Returns:
        float: angle to use for largest rectangle
    """

    min_distance = None
    closest_line = None

    # line is assigned below in the loop
    # lambda function sets the polygon and the line (which will be updated below)
    # and allows for x, y to be updated as needed
    partial_closest = lambda x, y: closest_line_string_in_poly(polygon, line, x, y)

    # update the line in the lambda to the new line and update new lines and distances
    # to find the global closest line and distance for the list of lines
    for line in linestrings:
        closest_line, min_distance = partial_closest(min_distance, closest_line)

    bearing = compass_bearing(closest_line)

    # Get the perpendicular angle and make it within 0 and 180°
    rectangleAngle = (bearing + 90) % 180

    return rectangleAngle

