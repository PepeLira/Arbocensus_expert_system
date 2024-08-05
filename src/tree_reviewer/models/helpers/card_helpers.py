import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt

def segment_intersection(p1, p2, q1, q2):
    """Returns the point of intersection between two line segments, if it exists, rounded to the precision of the input points."""
    def on_segment(p, q, r):
        return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
    
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2

    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    if o1 != o2 and o3 != o4:
        denom = (p1[0] - p2[0]) * (q1[1] - q2[1]) - (p1[1] - p2[1]) * (q1[0] - q2[0])
        px = ((p1[0] * p2[1] - p1[1] * p2[0]) * (q1[0] - q2[0]) - (p1[0] - p2[0]) * (q1[0] * q2[1] - q1[1] * q2[0])) / denom
        py = ((p1[0] * p2[1] - p1[1] * p2[0]) * (q1[1] - q2[1]) - (p1[1] - p2[1]) * (q1[0] * q2[1] - q1[1] * q2[0])) / denom
        return px, py
    
    if o1 == 0 and on_segment(p1, q1, p2):
        return q1
    if o2 == 0 and on_segment(p1, q2, p2):
        return q2
    if o3 == 0 and on_segment(q1, p1, q2):
        return p1
    if o4 == 0 and on_segment(q1, p2, q2):
        return p2

    return None

def plot_card(coords, intersection):
    fig, ax = plt.subplots()
    ax.plot(coords[:, 0], coords[:, 1], 'o-')
    ax.plot(intersection[0], intersection[1], 'ro')
    plt.show()

def find_single_intersecting_segments(coords):
    precision = 3
    coords = [tuple(np.round(coord, precision)) for coord in coords]
    indices = range(len(coords))
    segments = list(combinations(indices, 2))
    for (i1, j1), (i2, j2) in combinations(segments, 2):
        p1, p2 = coords[i1], coords[j1]
        q1, q2 = coords[i2], coords[j2]
        intersection = segment_intersection(p1, p2, q1, q2)
        if intersection:
            intersection = tuple(round(coord, precision) for coord in intersection)  # Round to the same precision as input points
            if intersection not in coords:
                return ([i1, j1], [i2, j2], intersection)
    
    return None

def get_card_sides(coords):
    # make a list with all posibles segments
    segments = []
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            segments.append([i, j])
    s1, s2, i = find_single_intersecting_segments(coords)
    for s in segments: # remove the segments that are the intersection
        if s == s1 or s == s2:
            segments.remove(s)
    
    return segments

def sides_length(coords, sides):
    sides_length = [(np.linalg.norm(coords[s[0]] - coords[s[1]]), s) for s in sides]
    # sort by length
    sides_length = sorted(sides_length, key=lambda x: x[0], reverse=True)
    return sides_length

def fix_orientation(coords):
    cp_coords = coords.copy()
    d_2 = np.linalg.norm(coords[1] - coords[2])
    d_3 = np.linalg.norm(coords[1] - coords[3])
    if d_2 < d_3:
        cp_coords[2], cp_coords[3] = coords[3], coords[2]
    return cp_coords

def sort_coords(coords):
    # round coords to 4 decimal places
    coords = np.round(coords, 4)
    # sort coords to be top left, top right, bottom right, bottom left
    # where the tl-tr and br-bl are the longest sides
    sides = get_card_sides(coords)
    sides_l = sides_length(coords, sides)
    sorted_coords = []
    for side in sides_l:
        for i in side[1]:
            if i not in sorted_coords:
                sorted_coords.append(i)
    return fix_orientation(coords[sorted_coords])