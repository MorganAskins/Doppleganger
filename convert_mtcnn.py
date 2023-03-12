import os
import sys
import time
from FaceUtilities import FaceUtilities

def convert_directory(dirA, dirB):
    '''
    Find the faces in dirA and move to dirB
    '''
    input_fnames = [f'{dirA}{v}' for v in os.listdir(dirA)]
    output_fnames = [f'{dirB}{v}' for v in os.listdir(dirA)]
    max_index = len(input_fnames)
    start_time = time.time()
    count = 1
    util = FaceUtilities()
    for index, (iname, oname) in enumerate(zip(input_fnames, output_fnames)):
        if os.path.exists(oname):
            continue
        try:
            extracted_image = util.extract_face(iname)
            extracted_image.save(oname)
            count += 1
        except:
            pass
        if count == 2:
            start_time = time.time()
        duration = time.time() - start_time
        time_per_image = duration / count
        remaining_images = max_index - index - 1
        remaining_time = remaining_images * time_per_image
        sys.stderr.write(f'Expected Duration ({index}/{max_index}): {remaining_time:0.0f} <{count}>           \r')
        if count > 1500:
            break

#convert_directory('data/CelebA/Img/', 'data/CelebA/ImgMTC/')
convert_directory('data/CelebA/TestImg/', 'data/CelebA/TestMTC/')
