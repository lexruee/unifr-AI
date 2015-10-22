#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import apa
import os


apa_data_path = "./apa"

nums = ['00', '01', '03', '04', '09', '10', '13', '14']
apa_dirs = ["%s/apa%s" % (apa_data_path, i) for i in nums]

def get_apa_files(apa_dirs):
    files = []
    for apa_dir in apa_dirs:
        files += map(lambda s: os.path.join(apa_dir, s), os.listdir(apa_dir))

    return files

files = get_apa_files(apa_dirs)

data, labels = apa.read_data_files(files)


print(labels)
print(data)