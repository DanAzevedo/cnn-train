import cv2

video = cv2.VideoCapture(0)

amostra = 1

while True:
    check, img = video.read()

    if cv2.waitKey(1) & 0xFF == ord('s'):
        imgR = cv2.resize(img, (32, 32))
        cv2.imwrite(f'Imagens/2/im{amostra}.jpg', imgR)
        print(f'imagem salva {amostra}')
        amostra += 1

    cv2.imshow('Captura', img)
    cv2.waitKey(1)
