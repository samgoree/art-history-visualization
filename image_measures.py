import numpy as np
import cv2
import scipy.stats

from io import BytesIO
import imageio
import sys
import scipy.signal

def log0(x):
    result = np.zeros_like(x)
    mask = (x != 0)
    result[mask] = np.log(x[mask])
    return result

def saturation_colorfulness(im):
    lab_1 = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
    return np.mean(np.sqrt(lab_1[:,:,1]**2 + lab_1[:,:,2]**2))

def variety_colorfulness(im):
    lab_1 = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
    mean = np.mean(lab_1, axis=(0,1))
    return np.mean(np.sqrt((lab_1[:,:,1] - mean[1])**2 + (lab_1[:,:,2] - mean[2])**2))

def lightness(im):
    lab_1 = cv2.cvtColor(im, cv2.COLOR_GTR2Lab)
    return np.mean(lab_1[:,:,0])

colorfulness_dict = {
    'saturation_colorfulness': saturation_colorfulness,
    'variety_colorfulness': variety_colorfulness,
    'lightness': lightness
}


def compression_complexity(im):
    uncompressed = im.nbytes
    buf = BytesIO()
    imageio.imwrite(buf, im, format='gif')
    return len(buf.getvalue())/uncompressed

def edge_complexity(im):
    im_saturation = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[:,:,1]
    edges = cv2.Canny(im_saturation, 200, 1,1)
    return np.std(edges)

def zipf_complexity(im):
    lightness = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)[:,:,0]
    hist, _ = np.histogram(lightness.flatten(), bins=256)
    sorted_hist = np.sort(hist)
    results = scipy.stats.linregress(log0(np.arange(0,256)), log0(sorted_hist))
    return results[0]

def machado_jpeg_complexity(sobel):
    original_size = im.nbytes
    buf = BytesIO()
    imageio.imwrite(buf, sobel, format='jpg')
    uncompressed_im = imageio.imread(buf.getvalue(), format='jpg')
    err = np.sqrt(np.mean((sobel - uncompressed_im)**2))
    return err * len(buf.getvalue())/original_size

def machado_zipf_rank_complexity(canny):
    hist, _ = np.histogram(canny.flatten(), bins=256)
    sorted_hist = np.sort(hist[np.nonzero(hist)])[::-1]
    
    results = scipy.stats.linregress(np.log(np.arange(1,len(sorted_hist) + 1)), np.log(sorted_hist))
    return results[0]

kernels = [
    np.array([
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, 0]]
    ),
    np.array([
        [0, 0, 0],
        [0, 1, -1],
        [0, 0, 0]]
    ),
    np.array([
        [0, 0, 0],
        [-1, 1, 0],
        [0, 0, 0]]
    ),
    np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, -1, 0]]
    )
]

def machado_zipf_size_complexity(canny):
    total_hist = np.zeros(256)
    for kernel in kernels:
        filtered_image = scipy.signal.convolve2d(canny, kernel)
        total_hist += np.histogram(np.abs(filtered_image), bins=256)[0]
    sorted_hist = np.sort(total_hist[np.nonzero(total_hist)])[::-1]
    
    results = scipy.stats.linregress(np.log(np.arange(1,len(sorted_hist) + 1)), np.log(sorted_hist))
    return results[0]

def machado_avg_complexity(canny):
    return np.mean(canny)

def machado_std_complexity(canny):
    return np.std(canny)

complexity_measures = {
    'machado_jpeg_complexity': machado_jpeg_complexity,
    'machado_zipf_rank_complexity': machado_zipf_rank_complexity,
    'machado_zipf_size_complexity': machado_zipf_size_complexity,
    'machado_avg_complexity': machado_avg_complexity,
    'machado_std_complexity': machado_std_complexity
}

measures = list(complexity_measures.values())
measure_names = list(complexity_measures.keys())


def compute_all_complexity_measures(im):
    im_saturation = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[:,:,1]
    canny = cv2.Canny(im_saturation, 200, 1)
    sobel = cv2.Sobel(im_saturation, cv2.CV_8U, 1, 1)
    results = []
    for i, measure in enumerate(measures):
        if measure_names[i] == 'machado_jpeg_complexity':
            results.append(measure(sobel))
        else:
            results.append(measure(canny))
    return np.array(results)

def colorfulness(im, measure='variety_colorfulness'):
    return colorfulness_dict[measure](im)

def complexity(im, measure='machado_avg_complexity'):
    im_saturation = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[:,:,1]
    if measure == 'machado_jpeg_complexity':
        sobel = cv2.Sobel(im_saturation, cv2.CV_8U, 1, 1)
        return complexity_measures[measure](sobel)
    else:
        canny = cv2.Canny(im_saturation, 200, 1)
        return complexity_measures[measure](canny)