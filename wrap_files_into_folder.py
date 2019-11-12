#! python3
# encoding = utf-8

import os, sys
import shutil

rootfolder = '/home/masha/HTQE/data/LMs_predict_quality/txt_elmo/'

for subdir, dirs, files in os.walk(rootfolder):
    # print('SUB', subdir)
    # print('DIR', dirs)
    # print(files)
    for i, file in enumerate(files):
        filepath = subdir + os.sep + file
        print('== CURRENT text:', filepath)
        outto = filepath.replace('.txt', '')
        print(outto)
        os.makedirs(outto, exist_ok=True)
        shutil.move(filepath, outto)
print('Done')