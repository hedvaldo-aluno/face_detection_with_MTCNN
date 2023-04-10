# Detecção de rostos utilizando Multi-task Cascaded Convolutional Networks (MTCNN)

## Resumo
O projeto em questão tem como principal objetivo fazer uso da inteligência artificial (IA) para identificar rostos em uma multidão capturada, como exemplo, por uma câmera de segurança.
## Introdução
A Multi-task Cascaded Convolutional Networks (MTCNN) é um algoritmo de detecção facial que foi proposto por Zhang et al. em 2016. Ela é usada para localizar e extrair características faciais em imagens, tais como detecção de rosto, detecção de pontos de referência faciais (como olhos, nariz e boca) e detecção de região de corte (bounding box) ao redor do rosto. Sendo composta por três redes neurais convolucionais em cascata: Rede de Detecção de Rosto (P-Net); Rede de Refinamento de Pontos de Referência Faciais (R-Net); Rede de Refinamento de Bounding Boxes (O-Net). O P-Net é a primeira rede na cascata e é responsável por realizar uma detecção rápida de regiões de possíveis rostos na imagem. Ela usa uma rede neural convolucional (CNN) para classificar regiões da imagem como "rosto" ou "não rosto" e também para ajustar as caixas em torno dos rostos detectados. 



## Código

```python
import cv2
from mtcnn import MTCNN
```
```python
cap = cv2.VideoCapture('Pessoas caminhando.mp4')
detector = MTCNN()

while True:

    ret,frame = cap.read()

    output = detector.detect_faces(frame)

    for single_output in output:
        x,y,width,height = single_output['box']
        cv2.rectangle(frame,pt1=(x,y),pt2=(x+width,y+height),color=(255,0,0),thickness=3)

    cv2.imshow('face detection',frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cv2.destroyAllWindows()
```


## Referências
https://medium.com/@lucas.kido19/detecção-facial-com-mtcnn-281c3707fe52

https://towardsdatascience.com/face-detection-using-mtcnn-a-guide-for-face-extraction-with-a-focus-on-speed-c6d59f82d49

https://www.lucaamore.com/?p=1143

https://pypi.org/project/mtcnn/

