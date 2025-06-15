# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:59:19 2021

@author: droes
"""
# You can use this library for oberserving keyboard presses
import keyboard # pip install keyboard

from capturing import VirtualCamera
from overlays import initialize_hist_figure, plot_overlay_to_image, plot_strings_to_image, update_histogram
from basics import histogram_figure_numba, mean_operation, mode_operation, std_operation, min_operation, max_operation, linear_transformation, entropy_operation, blur_filter, edge_detection_filter, equalization
import numpy as np
import cv2
import mediapipe as mp
import imageio


# Lip landmark indices (Face Mesh)
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_CORNER = 61
RIGHT_CORNER = 291

# Mouth Aspect Ratio
def mouth_aspect_ratio(landmarks, img_h, img_w):
    u = landmarks[UPPER_LIP]
    l = landmarks[LOWER_LIP]
    upper = np.array([u.x * img_w, u.y * img_h])
    lower = np.array([l.x * img_w, l.y * img_h])
    vert = np.linalg.norm(upper - lower)
    lc = landmarks[LEFT_CORNER]
    rc = landmarks[RIGHT_CORNER]
    left = np.array([lc.x * img_w, lc.y * img_h])
    right = np.array([rc.x * img_w, rc.y * img_h])
    horz = np.linalg.norm(left - right)
    return vert / horz if horz > 0 else 0

# Helper: overlays RGBA image onto BGR frame
def overlay_image(img, overlay, x, y):
    h, w = overlay.shape[:2]
    img_h, img_w = img.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, img_w), min(y + h, img_h)
    if x1 >= x2 or y1 >= y2:
        return img
    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)
    overlay_crop = overlay[oy1:oy2, ox1:ox2]
    b, g, r, a = cv2.split(overlay_crop)
    alpha = a.astype(float) / 255.0
    overlay_rgb = cv2.merge((r, g, b))
    roi = img[y1:y2, x1:x2]
    for c in range(3):
        roi[:, :, c] = (alpha * overlay_rgb[:, :, c] + (1 - alpha) * roi[:, :, c]).astype(np.uint8)
    img[y1:y2, x1:x2] = roi
    return img


def custom_processing(img_source_generator):
    # Tracker für Status Histogramm, Werte, Magier, Ice-Version
    values_on = False
    histogram_on = False
    mage_on = False
    ice_on = False

    # use this figure to plot your histogram
    fig, ax, background, r_plot, g_plot, b_plot = initialize_hist_figure()

    # Setup von MediaPipe Face Mesh
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

    # Import PNG (Magierhüte)
    hat_rgba = cv2.imread('hat.png', cv2.IMREAD_UNCHANGED)

    # Import GIFs fire/ice
    fire_reader = imageio.get_reader('fire.gif')
    fire_frames = [cv2.cvtColor(f, cv2.COLOR_RGBA2BGRA) for f in fire_reader]
    fire_reader.close()
    ice_reader = imageio.get_reader('ice_storm.gif')
    ice_frames = [cv2.cvtColor(f, cv2.COLOR_RGBA2BGRA) for f in ice_reader]
    ice_reader.close()
    n_fire = len(fire_frames)
    n_ice = len(ice_frames)

    # caches for performance improvements
    last_fire_w = None
    cached_fire = None
    last_ice_w = None
    cached_ice = None

    frame_idx = 0
    fire_idx = 0
    ice_idx = 0


    for sequence in img_source_generator:
        frame_idx += 1
        # Call your custom processing methods here! (e. g. filters)
        
        #Ohne Numpy Variante:
        #Linear Transformation
        #Linear Transformation auskommentieren, wenn man sie nicht braucht
        #seq_list = sequence.tolist()
        #seq_list = linear_transformation(seq_list, alpha=1.2, beta=70)
        #sequence = np.array(seq_list, dtype=np.uint8)

        #Filter
        #Filter auskommentieren wenn man sie nicht braucht
        #py = sequence.tolist()                           
        #py = blur_filter(py, size=15)                       
        #sequence = np.array(py, dtype=np.uint8) 
        #py = sequence.tolist()                           
        #py = edge_detection_filter(py)                       
        #sequence = np.array(py, dtype=np.uint8)

        #Histogram Equalization
        #Histogram Equalization auskommentieren, wenn man sie nicht braucht
        #py_eq = sequence.tolist()
        #py_eq = equalization(py_eq)
        #sequence = np.array(py_eq, dtype=np.uint8) 


        #Mit Numpy Variante:

        if keyboard.is_pressed('l') :   #Linear Transformation
            sequence = linear_transformation(sequence, alpha=1.2, beta=70)

        if keyboard.is_pressed('b') :      #Blur-Filter
            sequence = blur_filter(sequence, size=15)

        if keyboard.is_pressed('e') :       #Edge-Detection-Filter
            sequence = edge_detection_filter(sequence)

        if keyboard.is_pressed('z') :       #Equalization (Histogram)
            sequence = equalization(sequence)

        if keyboard.is_pressed('s') :       #Values (Mean, Mode, Max, ...)
            if values_on:
                values_on = False
            else:
                values_on = True

        if values_on:
            mean_vals = mean_operation(sequence)
            mode_vals = mode_operation(sequence)
            std_vals = std_operation(sequence)
            min_vals = min_operation(sequence)
            max_vals = max_operation(sequence)
            entropy_vals = entropy_operation(sequence)
            #Formatiere die Strings für den Overlay-Text
            stats_text = [
                f"Mean: R={mean_vals[2]:.1f}, G={mean_vals[1]:.1f}, B={mean_vals[0]:.1f}",
                f"Mode: R={mode_vals[2]}, G={mode_vals[1]}, B={mode_vals[0]}",
                f"Std : R={std_vals[2]:.1f}, G={std_vals[1]:.1f}, B={std_vals[0]:.1f}",
                f"Min : R={min_vals[2]}, G={min_vals[1]}, B={min_vals[0]}",
                f"Max : R={max_vals[2]}, G={max_vals[1]}, B={max_vals[0]}",
                f"Entr.: R={entropy_vals[2]:.2f}, G={entropy_vals[1]:.2f}, B={entropy_vals[0]:.2f}"
            ]
        
        #Histogram Logik
        if keyboard.is_pressed('h') :   
            if histogram_on:
                histogram_on = False
            else:
                histogram_on = True
        
        if histogram_on:
            # Load the histogram values
            r_bars, g_bars, b_bars = histogram_figure_numba(sequence)        
            
            # Update the histogram with new data
            update_histogram(fig, ax, background, r_plot, g_plot, b_plot, r_bars, g_bars, b_bars)
            
            # uses the figure to create the overlay
            sequence = plot_overlay_to_image(sequence, fig)
            

        ### Spezialaufgabe ###
        if keyboard.is_pressed('m') :
            print("Mage on/off")
            if mage_on:
                mage_on = False
            else:
                mage_on = True

        if keyboard.is_pressed('i') :
                print("Ice on/off")
                if ice_on:
                    ice_on = False
                else:
                    ice_on = True
        
        # Face and mouth detection overlay
        # Prepare RGB image for MediaPipe
        if mage_on:
            img_rgb = cv2.cvtColor(sequence, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img_rgb)
            h, w, _ = sequence.shape

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                # bounding box
                pts = np.array([[p.x * w, p.y * h] for p in lm])
                xmin, ymin = pts[:, 0].min(), pts[:, 1].min()
                xmax, ymax = pts[:, 0].max(), pts[:, 1].max()
                hairline_y = int(max(0, ymin - 0.1 * (ymax - ymin)))
                face_width = int(xmax - xmin)

                # mouth aspect
                mar = mouth_aspect_ratio(lm, h, w)
                mouth_open = mar > 0.5
                u = lm[UPPER_LIP]
                l = lm[LOWER_LIP]
                mouth_y = int((u.y + l.y) * 0.5 * h)
                left_x = int(lm[LEFT_CORNER].x * w)
                right_x = int(lm[RIGHT_CORNER].x * w)
                
                # mage hat cache
                if last_fire_w is None or abs(face_width - last_fire_w) > 10:
                    scale_h = (face_width * 2) / hat_rgba.shape[1]
                    new_w = int(hat_rgba.shape[1] * scale_h)
                    new_h = int(hat_rgba.shape[0] * scale_h)
                    cached_hat = cv2.resize(hat_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    last_fire_w = face_width  
                
                x_hat = int((xmin + xmax) / 2 - cached_hat.shape[1] / 2)
                y_hat = hairline_y +80 - cached_hat.shape[0]
                sequence = overlay_image(sequence, cached_hat, x_hat, y_hat)

                # Draw mouth effects only when open
                if mouth_open:
                    mouth_width = right_x - left_x

                    if ice_on:
                        # ICE cache (Ice Variante)
                        if last_ice_w is None or abs(mouth_width - last_ice_w) > 5:
                            f_h0, f_w0 = ice_frames[0].shape[:2]
                            scale_fi = (mouth_width / f_w0) * 1.5
                            fi_w, fi_h = int(f_w0 * scale_fi), int(f_h0 * scale_fi)
                            cached_ice = [cv2.resize(f, (fi_w, fi_h), cv2.INTER_AREA) for f in ice_frames]
                            last_ice_w = mouth_width
                        # advance index
                        if frame_idx % 3 == 0:
                            ice_idx = (ice_idx + 1) % n_ice
                        # overlay
                        frame_rgba = cached_ice[ice_idx]

                        # calculate position in mouth adapted to ice PNG
                        mouth_y = int((lm[UPPER_LIP].y + lm[LOWER_LIP].y) * 0.5 * h)
                        x_f = int(lm[LEFT_CORNER].x * w)
                        y_f = mouth_y
                    else:
                        # FIRE cache (Feuer Variante)
                        if last_fire_w is None or abs(mouth_width - last_fire_w) > 5:
                            f_h0, f_w0 = fire_frames[0].shape[:2]
                            scale_ff = (mouth_width * 3.5) / f_w0
                            ff_w, ff_h = int(f_w0 * scale_ff), int(f_h0 * scale_ff)
                            cached_fire = [cv2.resize(f, (ff_w, ff_h), cv2.INTER_AREA) for f in fire_frames]
                            last_fire_w = mouth_width
                        
                        # advance index
                        if frame_idx % 3 == 0:
                            fire_idx = (fire_idx + 1) % n_fire

                        # overlay
                        frame_rgba = cached_fire[fire_idx]

                        # calculate position in mouth adapted to fire PNG
                        mouth_y = int((lm[UPPER_LIP].y + lm[LOWER_LIP].y) * 0.5 * h)
                        x_f = int(lm[LEFT_CORNER].x * w) + 30
                        y_f = mouth_y -30

                    sequence = overlay_image(sequence, frame_rgba, x_f, y_f)
                    
        if values_on:
            sequence = plot_strings_to_image(sequence, stats_text)
        
        # Make sure to yield your processed image
        yield sequence



def main():
    # change according to your settings
    width = 1280
    height = 720
    fps = 30
    
    # Define your virtual camera
    vc = VirtualCamera(fps, width, height)
    
    vc.virtual_cam_interaction(
        custom_processing(
            # either camera stream
            vc.capture_cv_video(0, bgr_to_rgb=True)
            
            # or your window screen
            #vc.capture_screen()
        )
    )

if __name__ == "__main__":
    main()
