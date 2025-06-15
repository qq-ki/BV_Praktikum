# -*- coding: utf-8 -*-
"""
Created on Mon May  3 19:18:29 2021

@author: droes
"""
from numba import njit # conda install numba
import numpy as np
import math
import cv2

@njit
def histogram_figure_numba(np_img):
    '''
    Jit compiled function to increase performance.
    Use some loops insteads of purely numpy functions.
    If you face some compile errors using @njit, see: https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html
    In case you dont need performance boosts, remove the njit flag above the function
    Do not use cv2 functions together with @njit
    '''
    
    h, w, _ = np_img.shape
    #raw counts
    r_bars = np.zeros(256, dtype=np.float32)
    g_bars = np.zeros(256, dtype=np.float32)
    b_bars = np.zeros(256, dtype=np.float32)

    for y in range(h):
        for x in range(w):
            b = np_img[y, x, 0]
            g = np_img[y, x, 1]
            r = np_img[y, x, 2]
            b_bars[b] += 1.0
            g_bars[g] += 1.0
            r_bars[r] += 1.0

    #finde globalen max-Count über alle Kanäle
    max_c = 0.0
    for i in range(256):
        if b_bars[i] > max_c: max_c = b_bars[i]
        if g_bars[i] > max_c: max_c = g_bars[i]
        if r_bars[i] > max_c: max_c = r_bars[i]

    #skaliere so, dass max_c → 3.0 (damit es zur overlays.py funktion passt)
    if max_c > 0.0:
        scale = 3.0 / max_c
        for i in range(256):
            b_bars[i] *= scale
            g_bars[i] *= scale
            r_bars[i] *= scale

    #return r_bars, g_bars, b_bars
    return r_bars, g_bars, b_bars



####

### All other basic functions

####
'''
#Ohne numpy
def mean_operation(img):
    h = len(img)
    w = len(img[0])
    sums = [0.0, 0.0, 0.0]
    for y in range(h):
        for x in range(w):
            pixel = img[y][x]
            for c in range(3):
                sums[c] += pixel[c]
    count = h * w
    return [s / count for s in sums]

def min_operation(img):
    h = len(img)
    w = len(img[0])
    mins = list(img[0][0])
    for y in range(h):
        for x in range(w):
            pixel = img[y][x]
            for c in range(3):
                if pixel[c] < mins[c]:
                    mins[c] = pixel[c]
    return mins


def max_operation(img):
    h = len(img)
    w = len(img[0])
    maxs = list(img[0][0])
    for y in range(h):
        for x in range(w):
            pixel = img[y][x]
            for c in range(3):
                if pixel[c] > maxs[c]:
                    maxs[c] = pixel[c]
    return maxs


def std_operation(img):
    means = mean_operation(img)
    h = len(img)
    w = len(img[0])
    sum_sq = [0.0, 0.0, 0.0]
    for y in range(h):
        for x in range(w):
            pixel = img[y][x]
            for c in range(3):
                diff = pixel[c] - means[c]
                sum_sq[c] += diff * diff
    count = h * w
    variances = [s / count for s in sum_sq]
    return [math.sqrt(v) for v in variances]


def mode_operation(img):
    h = len(img)
    w = len(img[0])
    modes = []
    for c in range(3):
        freq = {}
        for y in range(h):
            for x in range(w):
                val = img[y][x][c]
                freq[val] = freq.get(val, 0) + 1
        mode_val = max(freq.items(), key=lambda kv: kv[1])[0]
        modes.append(mode_val)
    return modes


def entropy_operation(img):
    H = len(img)
    W = len(img[0])
    N = H * W

    #Histogramme initialisieren
    counts_b = [0] * 256
    counts_g = [0] * 256
    counts_r = [0] * 256

    for row in img:
        for (b, g, r) in row:
            counts_b[b] += 1
            counts_g[g] += 1
            counts_r[r] += 1

    entropies = []
    for counts in (counts_b, counts_g, counts_r):
        Hc = 0.0
        for cnt in counts:
            if cnt == 0:
                continue
            p = cnt / N
            Hc -= p * math.log2(p)
        entropies.append(Hc)

    return entropies 


def linear_transformation(img, alpha, beta):
    H = len(img)
    W = len(img[0])
    out = []
    
    for y in range(H):
        row_out = []
        for x in range(W):
            pixel = img[y][x]
            new_pix = []
            for c in range(3):
                val = int(pixel[c] * alpha + beta) #lineare Transformation, Alpha = Skalierungsfaktor(Kontrast), Beta=Offset(Helligkeit)
                #Werte auf (0,255) clippen
                if val < 0:
                    val = 0
                elif val > 255:
                    val = 255
                new_pix.append(val)
            row_out.append(tuple(new_pix))
        out.append(row_out)
    
    return out


#Filter
def blur_filter(img, size): #Low-Pass-Filter -> Blurring Effekt ("nur niedrige Frequenzen dürfen durch"), Box Blur Filter
    h = len(img)            #size: Kernelgröße (ungerade Zahl, z.B. 3,5,7…)
    w = len(img[0])
    r = size // 2
    out = [[(0,0,0) for _ in range(w)] for _ in range(h)]
    
    for y in range(h):
        for x in range(w):
            sum_b = sum_g = sum_r = 0
            count = 0
            for dy in range(-r, r+1):
                for dx in range(-r, r+1):
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < h and 0 <= nx < w:
                        pb, pg, pr = img[ny][nx]
                        sum_b += pb
                        sum_g += pg
                        sum_r += pr
                        count += 1
            #Mittelwert und Abrunden
            out[y][x] = (sum_b // count,
                         sum_g // count,
                         sum_r // count)

def edge_detection_filter(img):  #High-Pass-Filter -> Kantenerkennung (Edge Detection), "nur hohe Frequenzen dürfen durch"
    h = len(img)
    w = len(img[0])
    kernel = [
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1],
    ]
    out = [[(0,0,0) for _ in range(w)] for _ in range(h)]
    
    for y in range(h):
        for x in range(w):
            sum_b = sum_g = sum_r = 0
            for ky in range(3):
                for kx in range(3):
                    weight = kernel[ky][kx]
                    ny = y + (ky - 1)
                    nx = x + (kx - 1)
                    if 0 <= ny < h and 0 <= nx < w:
                        pb, pg, pr = img[ny][nx]
                        sum_b += pb * weight
                        sum_g += pg * weight
                        sum_r += pr * weight
            #auf [0,255] clippen
            def clamp(v):
                return 0 if v < 0 else 255 if v > 255 else v
            out[y][x] = (clamp(sum_b),
                         clamp(sum_g),
                         clamp(sum_r))
    return out

#Equalization
def equalization(img): #verteilt gleichmäßig die Intensitäten vom Bild über die komplette Breite vom Histogramm
    H = len(img)
    W = len(img[0])
    #Ausgabe initialisieren
    out = [[(0,0,0) for _ in range(W)] for _ in range(H)]

    #Für jeden der drei Kanäle separat
    for ch in range(3):
        #Histogramm zählen
        hist = [0]*256
        for y in range(H):
            for x in range(W):
                val = img[y][x][ch]
                hist[val] += 1

        #kumulative Verteilung
        cdf = [0]*256
        running = 0
        for i in range(256):
            running += hist[i]
            cdf[i] = running

        cdf_min = next(v for v in cdf if v>0)
        N = H*W

        #Lookup-Table bauen
        lut = [0]*256
        for i in range(256):
            lut[i] = int((cdf[i] - cdf_min) / (N - cdf_min) * 255 + 0.5)

        #neue Pixel
        for y in range(H):
            for x in range(W):
                pixel = list(out[y][x])
                pixel[ch] = lut[ img[y][x][ch] ]
                out[y][x] = tuple(pixel)

    return out

'''

#mit numpy
def mean_operation(img): 
    return img.mean(axis=(0, 1))


def std_operation(img): 
    return img.std(axis=(0, 1))


def min_operation(img):
    return img.min(axis=(0, 1))


def max_operation(img):
    return img.max(axis=(0, 1))


def mode_operation(img): 
    modes = []
    for c in range(img.shape[2]):
        flat = img[:, :, c].ravel()
        counts = np.bincount(flat, minlength=256)
        modes.append(int(counts.argmax()))
    return np.array(modes)

def entropy_operation(img):
    entropies = []
    for c in range(img.shape[2]):
        flat = img[:, :, c].ravel()
        #Histogramm der Grauwerte 0–255
        counts = np.bincount(flat, minlength=256).astype(np.float64)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        #Entropie in bit
        H = -np.sum(probs * np.log2(probs))
        entropies.append(H)
    return np.array(entropies) #Rückgabe: Entropy für jeden Farbkanal

def linear_transformation(img, alpha, beta):
    res = img.astype(np.float32) * alpha + beta #Alpha = Skalierungsfaktor(Kontrast), Beta=Offset(Helligkeit)
    res = np.clip(res, 0, 255) #Werte auf (0,255) clippen
    return res.astype(np.uint8)    


#Filter
def blur_filter(img, size): #Low-Pass-Filter -> Blurring Effekt ("nur niedrige Frequenzen dürfen durch"), Box Blur Filter
    #Kernel erzeugen
    kernel = np.ones((size, size), dtype=np.float32) / (size * size)
    #Filter anwenden
    return cv2.filter2D(img, ddepth=-1, kernel=kernel)

def edge_detection_filter(img):  #High-Pass-Filter -> Kantenerkennung (Edge Detection), "nur hohe Frequenzen dürfen durch"
    kernel = np.array([
        [ -1, -1,  -1],
        [-1,  8, -1],
        [ -1, -1,  -1]
    ], dtype=np.float32)
    return cv2.filter2D(img, ddepth=-1, kernel=kernel)

#Equalization
def equalization(img): #verteilt gleichmäßig die Intensitäten vom Bild über die komplette Breite vom Histogramm
    #Ausgabe initialisieren
    out = np.zeros_like(img, dtype=np.uint8)
    for ch in range(3):  #Für jeden Kanal einzeln:
        #Histogramm zählen
        flat = img[:, :, ch].ravel()
        hist = np.bincount(flat, minlength=256).astype(np.float32)
        
        #kumulative Verteilungsfunktion (CDF)
        cdf = hist.cumsum()
        #nur Werte >0 betrachten, damit cdf_min das erste nicht-null ist
        cdf_min = cdf[cdf > 0][0]
        n_pixels = flat.size
        
        #Lookup-Tabelle bauen
        #    → Werte in [0,255]
        lut = ((cdf - cdf_min) / (n_pixels - cdf_min) * 255.0).clip(0,255).astype(np.uint8)
        
        out[:, :, ch] = lut[img[:, :, ch]]
        
    return out