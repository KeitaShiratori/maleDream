#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import cv2
import numpy as np
import urllib.request
import io
from flask.ext.cors import CORS
from werkzeug import secure_filename


app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
app.debug = True

 
# This is the path to the upload directory
# app.config['UPLOAD_FOLDER'] = '/opt/app/flask/uploads/'
app.config['UPLOAD_FOLDER'] = 'D:\\dev\\pythonApp\\maleDream\\flask\\uploads\\'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']



@app.route('/api/v1/sample')
def index():
    data = {"result-code": "success"}

    img = cv2.imread(app.config['UPLOAD_FOLDER'] + 'sample.png')
    # グレースケール変換
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #Cannyアルゴリズムでエッジ検出
    edge_img = cv2.Canny(gray_img,100,200)

    flag, buf = cv2.imencode('.png', edge_img)
    return buf.tobytes()

## Cannyエッジ検出
@app.route('/cannyedge/<image>')
def canny(image=None):
    src = cv2.imread(app.config['UPLOAD_FOLDER'] + image)
    ## apply Canny Edge Detector
    return content

 
# Route that will process the file upload
@app.route('/api/v1/upload', methods=['POST'])
def upload():
    file = request.files['the_file']
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        # Move the file form the temporal folder to
        # the upload folder we setup
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Redirect the user to the uploaded_file route, which
        # will basicaly show on the browser the uploaded file
        return redirect(url_for('uploaded_file',
                                filename=filename))


# This route is expecting a parameter containing the name
# of a file. Then it will locate that file on the upload
# directory and show it on the browser, so if the user uploads
# an image, that image is going to be show after the upload
@app.route('/api/v1/uploads/<filename>')
def uploaded_file(filename):
    img = cv2.imread(app.config['UPLOAD_FOLDER']+filename)

    #カスケード分類器の特徴量を取得する
    cascade = cv2.CascadeClassifier("files/haarcascade_frontalface_alt.xml")

    # グレースケール変換
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #Cannyアルゴリズムでエッジ検出
    #edge_img = cv2.Canny(gray_img,100,200)

    #flag, buf = cv2.imencode('.jpg', edge_img)

    #物体認識（顔認識）の実行
    facerect = cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))

    # 囲む枠の色
    color = (255, 255, 255)

    # 認識した顔に対して線で囲む
    if len(facerect) > 0:
        #検出した顔を囲む矩形の作成
        for rect in facerect:
            print(rect)
            cv2.rectangle(img, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), color, thickness=2)

    # 肌色領域を切り取る
    img_skin = grabcut_skins(img, filename)

    # templatematch:
    img_path_array = [
                      '1ee673d185fc81436c8295f527ed2d5e.jpg',
                      'w700c-ez_37a77744fa88f68d4beaceadee672f87da17fc29ccf6bf5c.jpg',
                      'tumblr_mr50nhT9aB1qki7kio5_1280.jpg'
                     ]

    for path in img_path_array:
        try:
            gravia = cv2.imread('images/' + path, cv2.IMREAD_COLOR)     #比較対象画像
            face_recognize(gravia, 1)
            gravia2 = gravia.copy()
            gravia_gray = cv2.cvtColor(gravia, cv2.COLOR_RGB2GRAY)
            gravia2_gray = gravia_gray.copy()
            template = cv2.cvtColor(img_skin, cv2.COLOR_RGB2GRAY)     #テンプレート

            w, h = template.shape[::-1]

            # All the 6 methods for comparison in a list
            methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                       'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

            img_array = []

            for meth in methods:
                gravia = gravia2.copy()
                gravia_gray = gravia2_gray.copy()
                method = eval(meth)

            # Apply template Matching
            res = cv2.matchTemplate(gravia_gray, template, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)

            cv2.rectangle(gravia,top_left, bottom_right, 255, 2)
            img_array.append(gravia.copy())
            gravia_array = cv2.vconcat(img_array)

        except:
            print(path + ' errer!!')
            continue

    flag, buf = cv2.imencode('.jpg', gravia_array)
    return buf.tobytes()



def face_recognize(img, colorMode):
    if colorMode != 0:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = img

    #カスケード分類器の特徴量を取得する
    cascade = cv2.CascadeClassifier("files/haarcascade_frontalface_alt.xml")

    #物体認識（顔認識）の実行
    facerect = cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))

    # 囲む枠の色
    color = (255, 255, 255)

    # 認識した顔に対して線で囲む
    if len(facerect) > 0:
        #検出した顔を囲む矩形の作成
        for rect in facerect:
            print(rect)
            cv2.rectangle(img, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), color, thickness=2)



#@app.route('/api/v1/upload', methods=['POST'])
@app.route('/api/v1/templateMatch', methods=['POST'])
def template_match():
    file = request.files['rawData']
    #TODO ファイルデータの検証をここで行う
    #ファイル名を一意にするための処理もここで行う。ID使うとか。
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    img = cv2.imread(app.config['UPLOAD_FOLDER']+file.filename)
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    flag, buf = cv2.imencode('.png', img)
    return buf.tobytes()


# グラフカットで肌色領域を切り取る
def grabcut_skins(img, filename):
    thresh = 2.1
    thresh_ng = 0.6
    filename_out = (''.join(filename.split('.')[:-1])
                    + 'gc_out%d.' + filename.split('.')[-1])
    im_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    faces = detect_faces(img)
    face_hists = calc_face_histgram_normalized(img, faces, padding=20)
    contours = detect_contours(img)
    skin_contours = []
    skin_hists = []
    bg_contours = []

    for index, contour in enumerate(contours):
        hists = calc_contour_histgram_normalized(img, contour)
        for face_hist in face_hists:
            h_dist = cv2.compareHist(face_hist[0], hists[0], 0)
            s_dist = cv2.compareHist(face_hist[1], hists[1], 0)
            v_dist = cv2.compareHist(face_hist[2], hists[2], 0)
            if (h_dist + s_dist + v_dist) > thresh:
                skin_contours.append(contour)
            elif (h_dist + s_dist + v_dist) < thresh_ng:
                bg_contours.append(contour)

    skin_color = np.array([max_bin_histgram(face_hists[0][0]), max_bin_histgram(face_hists[0][1]), max_bin_histgram(face_hists[0][2])])
    diff_color = calc_hsv_diff(skin_color)

    # graph cut
    mask = mask_from_contours(img.shape[:2], skin_contours, bg_contours)
    print(mask)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    mask, bgd_model, fgd_model = cv2.grabCut(img, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    gc_mask = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    im_masked = img*gc_mask[:, :, np.newaxis]
    im_blown = blown_skin_mask(img, gc_mask, diff_color)

    im_concat = np.concatenate((img, im_masked), axis=1)

    return im_concat


def detect_faces(im):
    hc = cv2.CascadeClassifier("files/haarcascade_frontalface_alt.xml")
    faces = hc.detectMultiScale(im, minSize=(30, 30))
    if len(faces) == 0:
        raise Exception('no faces')
    return faces


def canny_edges(im):
    edge_im = cv2.Canny(im, 100, 200)
    kernel = np.ones((3, 3), np.uint8)
    opening_im = cv2.morphologyEx(edge_im, cv2.MORPH_CLOSE, kernel)
    return opening_im


def detect_contours(im):
    opening_im = canny_edges(im)
    ret, thresh = cv2.threshold(opening_im, 127, 255, 0)
    height, width = thresh.shape[:2]
    thresh[0:3, 0:width-1] = 255
    thresh[height-3:height-1, 0:width-1] = 255
    thresh[0:height-1, 0:3] = 255
    thresh[0:height-1, width-3:width-1] = 255
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    im_size = width * height
    ret = []
    for contour in contours:
        area =  cv2.contourArea(contour)
        if area < im_size/4:
            ret.append(contour)

    return ret


def calc_face_histgram_normalized(im, faces, padding=0):
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    hsv_histgrams = []
    for i, face in enumerate(faces):
        origin_x, origin_y, width, height = face
        area = (height - padding*2)*(width - padding*2)
        roi_image = im_hsv[(origin_y + padding):(origin_y + height - padding), (origin_x + padding):(origin_x + width - padding)]
        #cv2.imshow('win', roi_image)
        #cv2.waitKey(1000)

        h_h = cv2.calcHist([roi_image], [0], None, [180], [0, 180], None, 0)
        h_s = cv2.calcHist([roi_image], [1], None, [256], [0, 256], None, 0)
        h_v = cv2.calcHist([roi_image], [2], None, [256], [0, 256], None, 0)

        # normalization and append
        if area == 0:
            area = 1
        hsv_histgrams.append([h_h/area, h_s/area, h_v/area])

    return hsv_histgrams


def mask_from_contour(im_shape, contour):
    mask = np.zeros(im_shape, np.uint8)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    return mask

def mask_from_contours(im_shape, fg_contours, bg_contours):
    mask = np.zeros(im_shape, np.uint8)
    mask.fill(cv2.GC_PR_BGD)
    for contour in fg_contours:
        cv2.drawContours(mask, [contour], 0, cv2.GC_FGD, -1)
    for contour in bg_contours:
        cv2.drawContours(mask, [contour], 0, cv2.GC_BGD, -1)
    return mask


def calc_contour_histgram_normalized(im, contour):
    mask = mask_from_contour(im.shape, contour)
    area = cv2.contourArea(contour)
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    im_mask = cv2.inRange(mask, 10, 255)

    h_h = cv2.calcHist([im_hsv], [0], im_mask, [180], [0, 180], None, 0)
    h_s = cv2.calcHist([im_hsv], [1], im_mask, [256], [0, 256], None, 0)
    h_v = cv2.calcHist([im_hsv], [2], im_mask, [256], [0, 256], None, 0)

    # normalization and return
    if area == 0:
        area = 1
    return [h_h/area, h_s/area, h_v/area]


def blown_skin_mask(im, mask, diff_color):
    height, width = im.shape[:2]
    hsv_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    for y in range(height):
        for x in range(width):
            if (mask[y][x]):
                h = hsv_im[y, x, 0] + diff_color[0]
                s = hsv_im[y, x, 1] + diff_color[1]
                v = hsv_im[y, x, 2] + diff_color[2]
                if h < 0:
                    hsv_im[y, x, 0] = h + 180
                elif h > 180:
                    hsv_im[y, x, 0] = h - 180
                else:
                    hsv_im[y, x, 0] = h
                if s < 0:
                    hsv_im[y, x, 1] = 0
                elif s > 255:
                    hsv_im[y, x, 1] = 255
                else:
                    hsv_im[y, x, 1] = s
                if v < 0:
                    hsv_im[y, x, 2] = 0
                elif v > 255:
                    hsv_im[y, x, 2] = 255
                else:
                    hsv_im[y, x, 2] = v
    return cv2.cvtColor(hsv_im, cv2.COLOR_HSV2BGR)


#def calc_hsv_diff(skin_hist):
def calc_hsv_diff(skin_color):
    #blown_color = np.array([13, 110, 220])
    blown_color = np.array([3, 55, 110])
    return blown_color - skin_color


def max_bin_histgram(hist):
    idx, _ = np.unravel_index(hist.argmax(), hist.shape)
    return idx





if __name__ == '__main__':
    app.run(host='0.0.0.0')



