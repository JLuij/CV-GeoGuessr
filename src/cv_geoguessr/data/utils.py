from os import listdir
from os.path import isfile, join


def get_image_file_names_in_dir(dir: str):
    return [f for f in listdir(dir) if isfile(join(dir, f)) and f.endswith('.jpg')]


def take_coordinate_from_file_name(file_name: str):
    lat, long = file_name.split('_')

    # Remove .jpg
    long = long[:-4]

    return float(lat), float(long)
