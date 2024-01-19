import math, random
import albumentations as A

# source: https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon

def random_angle_steps(steps, irregularity):
    # generate n angle steps
    angles = []
    lower = (2 * math.pi / steps) - irregularity
    upper = (2 * math.pi / steps) + irregularity
    cumsum = 0
    for i in range(steps):
        angle = random.uniform(lower, upper)
        angles.append(angle)
        cumsum += angle

    # normalize the steps so that point 0 and point n+1 are the same
    cumsum /= (2 * math.pi)
    for i in range(steps):
        angles[i] /= cumsum
    return angles

def clip(value, lower, upper):
    return min(upper, max(value, lower))

def generate_polygon(center, avg_radius,
                     irregularity, spikiness,
                     num_vertices):
    """
    Start with the center of the polygon at center, then creates the
    polygon by sampling points on a circle around the center.
    Random noise is added by varying the angular spacing between
    sequential points, and by varying the radial distance of each
    point from the centre.

    Args:
        center (Tuple[float, float]):
            a pair representing the center of the circumference used
            to generate the polygon.
        avg_radius (float):
            the average radius (distance of each generated vertex to
            the center of the circumference) used to generate points
            with a normal distribution.
        irregularity (float):
            variance of the spacing of the angles between consecutive
            vertices.
        spikiness (float):
            variance of the distance of each vertex to the center of
            the circumference.
        num_vertices (int):
            the number of vertices of the polygon.
    Returns:
        List[Tuple[float, float]]: list of vertices, in CCW order.
    """
    # Parameter check
    if irregularity < 0 or irregularity > 1:
        raise ValueError("Irregularity must be between 0 and 1.")
    if spikiness < 0 or spikiness > 1:
        raise ValueError("Spikiness must be between 0 and 1.")

    irregularity *= 2 * math.pi / num_vertices
    spikiness *= avg_radius
    angle_steps = random_angle_steps(num_vertices, irregularity)

    # now generate the points
    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in range(num_vertices):
        radius = clip(random.gauss(avg_radius, spikiness), 0, 2 * avg_radius)
        point = (center[0] + radius * math.cos(angle),
                 center[1] + radius * math.sin(angle))
        points.append(point)
        angle += angle_steps[i]

    return points

def get_train_augmentations():
    base_augmentations = [
        A.CropNonEmptyMaskIfExists(height=384, width=640),
        A.Rotate(limit=20),
        A.HorizontalFlip(p=0.5),
    ]
    right_augmentations = [
        A.GaussNoise(p=0.3),
        # A.InvertImg(p=0.1),
        A.RandomShadow(),
        A.RandomSunFlare(src_radius=200, p=0.1),
        A.RandomRain(p=0.1),
        # A.RandomSnow(p=0.1),
    ]
    disp_augmentations = [
        A.GaussNoise(var_limit=0.005, p=1)
    ]

    base_aug = A.Compose(base_augmentations,
        additional_targets={
            'gt_disp': 'image',}
        )
    real_aug = A.Compose(base_augmentations,
        additional_targets={
            'gt_disp': 'image',
            'disp': 'image'}
        )

    right_aug = A.Compose(right_augmentations)

    disp_aug = A.Compose(disp_augmentations)

    return base_aug, right_aug, disp_aug, real_aug
