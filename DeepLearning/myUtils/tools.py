# -*- coding = utf-8 -*-
# Author: Shaochi(KesarChi) Wang
# Institute: Beijing Institute Of Genomics, CAS.


import os


def get_all_picnames_of_path(path):
    files = os.listdir(path)
    files = [f for f in files if f.endswith('.png') or f.endswith('.jpg')]
    return files


def get_all_absPicNames_of_path(path):
    files = get_all_picnames_of_path(path)
    files = [os.path.join(path, f) for f in files]
    return files