import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import pycuda.autoinit
import pycuda.driver as driver
from pycuda.compiler import SourceModule

def generateDataset(videofile, buffer_size, scale, output_path, simulatrity_value,
                    start_clip=0, end_clip=0, save_index=0):
    """[summary]

    Args:
        videofile ([string]): [path to video file]
        buffer_size ([int]): [number of images to load into RAM]
        scale ([float]): [factor by which to downscale image]
        output_path ([string]): [path to save image files]
        simulatrity_value ([float]): [determines to keep frame or not]
        start_clip ([int]): number of frames to clip from the start
        end_clip: number of frames to clip from the end
        save_index: pad the name of the file 
    """
    with open('./source/cuda_kernels.cu', 'r') as file:
        source = file.read()
        __module = SourceModule(source)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    video = cv2.VideoCapture(videofile)
    if video.isOpened():
        total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        video.set(cv2.CAP_PROP_POS_FRAMES, start_clip)
    else:
        return 0

    BATCHES = int((total_frames - start_clip - end_clip) // buffer_size)
    THREADCOUNT = 8
    remainder = (total_frames - start_clip - end_clip) % buffer_size
    total_files = 0

    for batch in range(0, BATCHES+1):
        print(f'\nBatch {batch+1} of {BATCHES+1}')
        print("Loading files:")
        frames = []
        if batch == BATCHES: buffer_size = int(remainder)
        for buffer in tqdm(range(buffer_size)):
            hasFrame, frame = video.read()
            if not hasFrame:
                break
            h, w = frame.shape[:2]
            dim = (int(w//scale), int(h//scale))
            frame = np.array(cv2.resize(frame, dim)).astype(np.uint8)
            frames.append(frame)

        frames_device = []
        print("Copying to GPU:")
        for i in tqdm(range(0, len(frames))):
            frames_device.append(driver.mem_alloc(frames[i].nbytes))
            driver.memcpy_htod(frames_device[i], frames[i])
        
        diffBlock = (8,8,3)
        diffGrid = (int(dim[0] / 8), int(dim[1] / 8), 1)
        h_diffImage_int = np.zeros_like(frames[0], dtype=np.uint8)
        d_diffImage_int = driver.mem_alloc(h_diffImage_int.nbytes)
        getImgDiff = __module.get_function("cuda_GetImgDiff")

        num_block = int(dim[0] * dim[1] * 3 / 512)
        block = (512,1,1)
        grid = (num_block,1,1)

        h_sum = np.zeros(num_block, dtype=np.float32)
        d_sum = driver.mem_alloc(h_sum.nbytes)
        sumPixels = __module.get_function("cuda_SumPixels")

        h_diffImage_float = h_diffImage_int.astype(np.float32)
        d_diffImage_float = driver.mem_alloc(h_diffImage_float.nbytes)
        byteToFloat = __module.get_function("cuda_ByteToFloat")

        print("Processing")
        pbar = tqdm()
        index = 0
        while(True):
            if index + 1 >= len(frames):
                break
            getImgDiff(d_diffImage_int, frames_device[index],
                        frames_device[index+1],
                        np.int32(dim[0]), block=diffBlock, grid=diffGrid)
            driver.memcpy_dtoh(h_diffImage_int, d_diffImage_int)
            byteToFloat(d_diffImage_float, d_diffImage_int, block=block, grid=grid)
            sumPixels(d_diffImage_float, d_sum, block=block, grid=grid)
            driver.memcpy_dtoh(h_sum, d_sum)
            mse = h_sum.sum() / (dim[0] * dim[1])
            if mse < simulatrity_value:
                del frames[index+1]
                del frames_device[index+1]
            else:
                index += 1
            pbar.update(1)
        print("Saving")
        for i in tqdm(range(0, len(frames))):
            cv2.imwrite(output_path + f"{save_index + total_files + i:05d}.jpg", frames[i])
        total_files += len(frames)
        print(f"Done:\tKept {len(frames)} frames")

    os.system('clear')
    return total_files
        # test_int = 300
        # delta_int = 1
        # getImgDiff(d_diffImage_int, frames_device[test_int], frames_device[test_int+delta_int], np.int32(dim[0]), block=diffBlock, grid=diffGrid)
        # driver.memcpy_dtoh(h_diffImage_int, d_diffImage_int)
        # img = Image.fromarray(h_diffImage_int, 'RGB')
        # img.save('diff.png')
        # img = Image.fromarray(frames[test_int], 'RGB')
        # img.save('0.png')
        # img = Image.fromarray(frames[test_int+delta_int], 'RGB')
        # img.save('1.png')
        # byteToFloat(d_diffImage_float, d_diffImage_int, block=block, grid=grid)
        # # driver.memcpy_dtoh(h_diffImage_float, d_diffImage_float)
        # sumPixels(d_diffImage_float, d_sum, block=block, grid=grid)
        # driver.memcpy_dtoh(h_sum, d_sum) # pylint: disable=no-member
        # pixelSum = h_sum.sum()
        # print(pixelSum/(dim[0]*dim[1]))
        

    