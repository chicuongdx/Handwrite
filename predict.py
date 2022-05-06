import pickle
import matplotlib.pyplot as plt
import cv2

filename = 'model_svm.sav'
clf = pickle.load(open(filename, 'rb'))

def show2image(img1, img2, title1, title2):
    f = plt.figure(figsize = (15, 15))
    f.add_subplot(1,2, 1)
    plt.title(title1)
    imgLr = cv2.cvtColor(imgLr, cv2.COLOR_BGR2RGB)
    plt.imshow(imgLr)
    f.add_subplot(1,2, 2)
    plt.title(title2)
    imgRr = cv2.cvtColor(imgRr, cv2.COLOR_BGR2RGB)
    plt.imshow(imgRr)
    plt.show(block=True)

def showImages_1line( imgs, titles ):
    f = plt.figure(figsize = (15, 15))
    if len(imgs) != len(titles):
        print( "not same amounts")
        return

    leng = len(imgs)
        
    for i in range( len(imgs)):
        f.add_subplot(1,leng, i+1)
        plt.title( titles[i])
        img = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
        plt.imshow(img)
    
    plt.show()


def load_and_predict(path):
    # LOAD ẢNH SAU KHI CHỤP BẰNG WEBCAM
    _image = cv2.imread(path, 0)

    # TOP, BOT, LEFT, BOT ĐANG LÀ TỌA ĐỘ CỦA ĐIỂM GIỮA TẤM ẢNH
    # CÁC BIẾN NÀY CẦN TÍNH TOÀN THÀNH CÁC CỰC ĐỘ CỦA CHỮ SỐ TRONG TẤM ẢNH
    top   = _image.shape[0]//2
    bot   = _image.shape[0]//2
    left  = _image.shape[1]//2
    right = _image.shape[1]//2

    # TIỀN XỬ LÝ
    # XỬ LÝ PHẦN SÁNG VÀ TỐI TRONG BỨC ẢNH ĐỂ CẮT ẢNH
    # ẢNH SAU KHI CẮT LÀ MỘT ẢNH HỈNH VUÔNG CHỨA CHỮ SỐ
    # BIẾN DARK VÀ BRIGHT TÙY VÀO BỨC ẢNH CÓ THỂ CHỈNH SỬA CHO PHÙ HỢP, NGƯỜI CODE CHƯA BIẾT CÁCH CHỈNH BẰNG CODE NÊN ĐÀNH CHỈNH BẰNG TAY
    DARK = 50
    BRIGHT = 120
    strictedImg = _image.copy()
    # TÍNH TOÁN CÁC CỰC ĐỘ CỦA CHỮ SỐ DỰA VÀO PHẦN MÀU DEN
    for xx in range(_image.shape[0]):
        for yy in range(_image.shape[1]):
            if _image[xx][yy] <= DARK:
                if top >= xx: top = xx
                if bot <= xx: bot = xx
                if left >= yy: left = yy
                if right <= yy: right = yy

                strictedImg[xx][yy] = 0
            if _image[xx][yy] > BRIGHT:
                strictedImg[xx][yy]=255

    # DỰA VÀO CÁC 
    edge = bot - top
    if right - left > edge:
        edge = right - left
    edge = edge // 2
    midx = ( bot + top ) // 2
    midy = ( left + right ) // 2
    strictedImg = strictedImg[ midx-edge:midx+edge, midy-edge:midy+edge ]
    strictedImg = 255 - strictedImg[:,:]

    # CHỈNH ẢNH VỀ FORMAT CÓ THỂ ĐƯA VÀO MODEL
    # MODEL CÓ INPUT LÀ ẢNH 8X8, MỖI PIXEL CÓ GIÁ TRỊ THUỘC [0,16]
    strictedImg_resized = cv2.resize( strictedImg, dsize=( 8,8 ) )
    strictedImg_resized = strictedImg_resized[:,:] // 16

    # IN RA ẢNH SAU KHI QUA BƯỚC TIỀN XỬ LÝ
    # DỰ ĐOÁN VÀ ĐƯA RA KẾT QUẢ
    _result = clf.predict( strictedImg_resized.reshape( (1,64) ) )
    #print('Ảnh đã qua tiền xử lý : ')
    #plt.imshow( strictedImg ) 
    #print('Ảnh đã resize về 8x8')
    #plt.imshow( cv2.resize( strictedImg_resized, dsize = (edge * 2, edge * 2) ))
    print('predict : ',_result[0])
    
    resized_image = cv2.resize( strictedImg_resized, dsize = (edge * 2, edge * 2) )
    showImages_1line( [_image, strictedImg, resized_image], [ "gốc","qua tiền xử lý","resize 8x8"])

load_and_predict("7.jpg")