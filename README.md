# Detecção de rostos utilizando Multi-task Cascaded Convolutional Networks (MTCNN)


## Resumo
O projeto em questão tem como principal objetivo fazer uso da inteligência artificial (IA) para identificar rostos em uma multidão capturada, como exemplo, por uma câmera de segurança.
## Introdução
A Multi-task Cascaded Convolutional Networks (MTCNN) é um algoritmo de detecção facial que foi proposto por Zhang et al. em 2016. Ela é usada para localizar e extrair características faciais em imagens, tais como detecção de rosto, detecção de pontos de referência faciais (como olhos, nariz e boca) e detecção de região de corte (bounding box) ao redor do rosto. Sendo composta por três redes neurais convolucionais em cascata: Rede de Detecção de Rosto (P-Net); Rede de Refinamento de Pontos de Referência Faciais (R-Net); Rede de Refinamento de Bounding Boxes (O-Net). O P-Net é a primeira rede na cascata e é responsável por realizar uma detecção rápida de regiões de possíveis rostos na imagem. Ela usa uma rede neural convolucional (CNN) para classificar regiões da imagem como "rosto" ou "não rosto" e também para ajustar as caixas em torno dos rostos detectados. Depois que o P-Net detecta possíveis regiões de rosto, a R-Net é usada para refinar a detecção, ajustando as caixas e também detectando os pontos de referência faciais. A terceira rede na cascata é a O-Net, que é responsável por refinar ainda mais as caixas e fornecer uma detecção final mais precisa dos pontos de referência faciais. Assim como as redes anteriores, a O-Net também é uma CNN que realiza classificação e regressão para ajustar as caixas e predizer os pontos de referência faciais finais.

## Código

![2023-04-10-20-05-10](https://user-images.githubusercontent.com/113546603/231015819-d9247048-6585-4b85-bec6-a043919f135e.gif)

### Código colab(.ipynb)
https://colab.research.google.com/drive/1yFYLluGYOIub-zmPWsaot7OeBw89Bn5r?usp=sharing

### Código python(.py)
```python
# Requerimentos:
# mtcnn==0.1.1
# opencv-python==4.7.0.72
import cv2
from mtcnn import MTCNN
```
```python
# Usado para capturar vídeo de uma fonte de entrada, como uma câmera de vídeo ou um arquivo de vídeo.
cap = cv2.VideoCapture('Pessoas caminhando.mp4')
# Cria um objeto de detecção de faces, que pode ser usado para detectar faces em imagens.
detector = MTCNN()

while True:
    # Ler cada quadro da câmera/vídeo
    ret,frame = cap.read()
    # Função para detectar as faces na imagem
    output = detector.detect_faces(frame)

    for single_output in output:
        # Extrai as informações de caixa delimitadora (box) do dicionário single_output e as atribui às variáveis x, y, width e height. 
        x,y,width,height = single_output['box']
        # Desenha um retângulo ao redor da face detectada na imagem
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

