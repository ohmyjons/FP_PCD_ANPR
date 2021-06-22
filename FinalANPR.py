# from PlatRecog import lowlight
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
import imutils


# def Anpr_Indo(img):

# def __init__(self,img):
#     self.img = img
# PRAPENGOLAHAN

# load citra RGB (BGR)
# img = cv.imread("./test_images/1.jpg") # plat nomer not detect
# img = cv.imread("./test_images/2.jpg") # salah segmentasi platnomer
# img = cv.imread("./test_images/09.jpg") #salah segmentasi karakter
# img = cv.imread("./test_images/9.jpg") 
# img = cv.imread("./test_images/10.jpg") 
# img = cv.imread("./test_images/123.jpg")  #salah segmentasi platnomer
# img = cv.imread("./test_images/124.jpg") #segmentasi plat salah
# img = cv.imread("./test_images/AA5627JT.jpg")
# img = cv.imread("./test_images/AB2638XU.jpg") 
# img = cv.imread("./test_images/AB5592EG.jpg")
img = cv.imread("./test_images/AD2914JG.jpg")  # segmentasi salah D dobel
# img = cv.imread("./test_images/B3023KEZ.jpg")
# img = cv.imread("./test_images/plat1.jpeg") 
# img = cv.imread("./test_images/plat2.jpeg") 
# img = cv.imread("./test_images/plat3.jpeg") 
# img = cv.imread("./test_images/platw.jpeg")  #Salah Segmentasi 1 dan 0
# img = cv.imread("./test_images/platad.jpeg")  # Salah Segementasi di D
# img = cv.imread("./test_images/platl.jpeg") 
# img = cv.imread("./test_images/plat3.jpeg") 
# img = cv.imread("./test_images/plat_new_1.jpeg") 
cv.imshow('img',img)
cv.waitKey(0)



# resize citra dengan imutils
# dimana setiap citrayang masuk akan diubah ukuran pixelnya menjadi 1280
# degnan ratio tetap sama
img = imutils.resize(img, width = 1280)

# konversi dari RGB ke grayscale
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cek_lowlight = 0

# Normalisasi Cahaya
def normalisasiCahaya(img_gray):
    # buat kernel dengan bentuk ellipse, diameter 20 piksel
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(20,20))

    # lakukan operasi opening ke citra grayscale dengan kernel yang sudah dibuat (var: kernel)
    img_opening = cv.morphologyEx(img_gray, cv.MORPH_OPEN, kernel)

    # Normalisasi cahaya (2)
    # lakukan pengurangan citra grayscale dengan citra hasil opening
    img_norm = img_gray - img_opening

    # Normalisasi cahaya (3)
    # konversi citra hasil normalisasi ke citra BW (hitam putih)
    # (thresh, img_norm_bw) = cv.threshold(img_norm, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    (thresh, img_norm_bw) = cv.threshold(img_norm, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # ==== Cek normalisasi START ====
    # Untuk ngecek hasil sebelum dan sesudah dilakukan normalisasi
    # Bisa di comment/uncomment

    # buat citra bw tanpa normalisasi
    (thresh, img_without_norm_bw) = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    fig = plt.figure(figsize=(10, 7))
    row_fig = 2
    column_fig = 2

    fig.add_subplot(row_fig, column_fig, 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.axis('on')
    plt.title("RGB")

    fig.add_subplot(row_fig, column_fig, 2)
    plt.imshow(img_gray, cmap='gray')
    plt.axis('on')
    plt.title("Grayscale")

    fig.add_subplot(row_fig, column_fig, 3)
    plt.imshow(img_without_norm_bw, cmap='gray')
    plt.axis('on')
    plt.title("Tanpa Normalisasi")

    fig.add_subplot(row_fig, column_fig, 4)
    plt.imshow(img_norm_bw, cmap='gray')
    plt.axis('on')
    plt.title("Dengan Normalisasi")

    plt.show()

    return img_norm_bw

# apabila cahaya lowlight
def lowLight(img_grey):
    # Tophat
    kernel = np.ones((20,20), np.uint8)
    tophat = cv.morphologyEx(img_grey,cv.MORPH_TOPHAT, kernel)
    
    # otsu + biner image
    ret,img_otsu = cv.threshold(tophat,0,255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    cv.imshow('Otsu',img_otsu)    
    
    fig = plt.figure(figsize=(10, 7))
    row_fig = 2
    column_fig = 2
    fig.add_subplot(row_fig, column_fig, 1)
    plt.imshow(cv.cvtColor(tophat, cv.COLOR_BGR2RGB))
    plt.axis('on')
    plt.title("tophat")

    fig.add_subplot(row_fig, column_fig, 2)
    plt.imshow(img_otsu, cmap='gray')
    plt.axis('on')
    plt.title("otsu")

    plt.show()
    return img_otsu


# def dilations(img):



# call fungsi normalisasi
img_norm_bw = normalisasiCahaya(img_gray)
# img_lowlight = lowLight(img_gray)
# img_norm_bw = lowLight(img_gray)


# Deteksi Platnomer
# Dalam penerapannya kita menggunakan cotours dari open cv
# untuk mendeteksi plat nomer

def deteksiPlatnomer(img_norm_bw,img_gray):
    global img_plate_gray
    global cek_lowlight
    global img_show_plate,x_plate,y_plate,h_plate
    # dapatkan contours dari citra kendaraan
    contours_vehicle, hierarchy = cv.findContours(img_norm_bw, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) # get the contour for every area

    # cek jumlah contours
    # print(len(contours_vehicle))

    # index contour yang berisi kandidat plat nomor
    index_plate_candidate = []

    # index counter dari setiap contour di contours_vehichle
    index_counter_contour_vehicle = 0

    # filter setiap contour untuk mendapatkan kandidat plat nomor
    for contour_vehicle in contours_vehicle:
        
        # dapatkan posisi x, y, nilai width, height, dari contour
        x,y,w,h = cv.boundingRect(contour_vehicle)

        # dapatkan nilai aspect rationya
        aspect_ratio = w/h

        # dapatkan kandidat plat nomornya apabila:
        # 1. lebar piksel lebih dari atau sama dengan 200 piksel
        # 2. aspect rationya kurang dari atau sama dengan 4
        if w >= 200 and aspect_ratio <= 4 : 
            
            # dapatkan index kandidat plat nomornya
            index_plate_candidate.append(index_counter_contour_vehicle)
        
        # increment index counter dari contour
        index_counter_contour_vehicle += 1

    # buat duplikat citra RGB dan BW kendaraan untuk menampilkan lokasi plat
    img_show_plate = img.copy() 
    img_show_plate_bw = cv.cvtColor(img_norm_bw, cv.COLOR_GRAY2RGB)
    # cv.imshow('img_bw',img_show_plate_bw)
    # cv.waitKey(0)

    # print len(index_plate_candidate)
    if len(index_plate_candidate) == 0:

        # tampilkan peringatan plat nomor tidak terdeteksi
        print("Plat nomor tidak ditemukan")
        cek_lowlight = 1
        print(cek_lowlight)
        deteksiPlatnomer(lowLight(img_gray),img_gray)

    # jika jumlah kandidat plat sama dengan 1
    elif len(index_plate_candidate) == 1:

        # dapatkan lokasi untuk pemotongan citra plat
        x_plate,y_plate,w_plate,h_plate = cv.boundingRect(contours_vehicle[index_plate_candidate[0]])
        
        # gambar kotak lokasi plat nomor di citra RGB
        cv.rectangle(img_show_plate,(x_plate,y_plate),(x_plate+w_plate,y_plate+h_plate),(0,255,0),5)

        # gambar kotak lokasi plat nomor di citra BW
        cv.rectangle(img_show_plate_bw,(x_plate,y_plate),(x_plate+w_plate,y_plate+h_plate),(0,255,0),5)

        # crop citra plat 
        img_plate_gray = img_gray[y_plate:y_plate+h_plate, x_plate:x_plate+w_plate]
        print(img_plate_gray)
    else:
        print('Dapat dua lokasi plat, pilih lokasi plat kedua')

        # dapatkan lokasi untuk pemotongan citra plat
        x_plate,y_plate,w_plate,h_plate = cv.boundingRect(contours_vehicle[index_plate_candidate[1]])

        # gambar kotak lokasi plat nomor di citra RGB
        cv.rectangle(img_show_plate,(x_plate,y_plate),(x_plate+w_plate,y_plate+h_plate),(0,255,0),5)

        # gambar kotak lokasi plat nomor di citra BW
        cv.rectangle(img_show_plate_bw,(x_plate,y_plate),(x_plate+w_plate,y_plate+h_plate),(0,255,0),5)

        # crop citra plat 
        img_plate_gray = img_gray[y_plate:y_plate+h_plate, x_plate:x_plate+w_plate]
        # print(img_plate_gray)

    # ==== Cek Deteksi Plat START ====
    # Bisa di comment/uncomment

    fig2 = plt.figure(figsize=(10, 7))
    row_fig = 2
    column_fig = 2

    fig2.add_subplot(row_fig, column_fig, 1)
    plt.imshow(cv.cvtColor(img_show_plate_bw, cv.COLOR_BGR2RGB))
    plt.axis('on')
    plt.title("Lokasi Plat Nomor BW")

    fig2.add_subplot(row_fig, column_fig, 2)
    plt.imshow(cv.cvtColor(img_show_plate, cv.COLOR_BGR2RGB))
    plt.axis('on')
    plt.title("Lokasi Plat Nomor")

    fig2.add_subplot(row_fig, column_fig, 3)
    plt.imshow(img_plate_gray, cmap="gray")
    plt.axis('on')
    plt.title("Hasil Crop Plat Nomor")

    plt.show()

    return img_plate_gray


img_crop = deteksiPlatnomer(img_norm_bw,img_gray)
print(cek_lowlight)
# print(img_crop)

# SEGMENTASI KARAKTER
# karakter yang di segmentasi adalah baris pertama yang berisi nilai unik setiap kendaraan
def segmentasiKarakter(img_plate_gray):
    
    if cek_lowlight == 0 :
        # konversi dari grayscale ke BW
        (thresh, img_plate_bw) = cv.threshold(img_plate_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # hasil dari konversi BW tidak terlalu mulus, 
        # ada bagian-bagian kecil yang tidak diinginkan yang mungkin bisa mengganggu
        # maka hilangkan area yang tidak diinginkan dengan operasi opening

        # buat kernel dengan bentuk cross dan ukuran 3x3
        kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))

        # cv.imshow("sebelum open", img_plate_bw)

        # lakukan operasi opening dengan kernel di atas
        img_plate_bw = cv.morphologyEx(img_plate_bw, cv.MORPH_OPEN, kernel) # apply morph open
    
    elif cek_lowlight == 1 :
        img_plate_bw = lowLight(img_plate_gray)
        # img_plate_bw = dilations(img_plate_bw) 

    # cv.imshow("sesudah open", img_plate_bw)

    # Segmentasi karakter menggunakan contours
    # dapatkan kontur dari plat nomor
    contours_plate, hierarchy = cv.findContours(img_plate_bw, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) 

    # index contour yang berisi kandidat karakter
    index_chars_candidate = [] #index

    # index counter dari setiap contour di contours_plate
    index_counter_contour_plate = 0 #idx

    # duplikat dan ubah citra plat dari gray dan bw ke rgb untuk menampilkan kotak karakter
    img_plate_rgb = cv.cvtColor(img_plate_gray,cv.COLOR_GRAY2BGR)
    # cv.imshow('dsadsa',img_plate_rgb)


    img_plate_bw_rgb = cv.cvtColor(img_plate_bw, cv.COLOR_GRAY2RGB)
    # cv.imshow('plate_bw_segment',img_plate_bw_rgb)
    # cv.imshow('contur',contours_plate)
    # print(contours_plate)

    
    return 

# segmentasiKarakter(img_crop)
