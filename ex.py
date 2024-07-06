# STEP 1: 패키지 불러오기
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# STEP 2: 추론기 생성
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640)) # ctx_id=0 이면 GPU 가 아닌 CPU 를 씀

# STEP 3: 데이터 가져오기
img1 = cv2.imread("nohongcheol1.jpg")
img2 = cv2.imread("nohongcheol2.jpg")
img3 = cv2.imread("nohongcheol3.jpg")
img5 = cv2.imread("nohongcheol5.png")
img6 = cv2.imread("nohongcheol6.jpg")
img7 = cv2.imread("nohongcheol7.jpg")

# STEP 4: 추론
faces1 = app.get(img1)
faces2 = app.get(img2)
faces3 = app.get(img3)
faces5 = app.get(img5)
faces6 = app.get(img6)
faces7 = app.get(img7)

assert len(faces1)==1
assert len(faces2)==1
assert len(faces3)==1
# assert len(faces5)==1


print(faces1[0])
print(faces2[0])

# STEP 5: 활용
rimg1 = app.draw_on(img1, faces1)
cv2.imwrite("./nohongcheol1_output.jpg", rimg1)

rimg2 = app.draw_on(img2, faces2)
cv2.imwrite("./nohongcheol2_output.jpg", rimg2)

rimg5 = app.draw_on(img5, faces5)
cv2.imwrite("./nohongcheol5_output.jpg", rimg5)

rimg6 = app.draw_on(img6, faces6)
cv2.imwrite("./nohongcheol6_output.jpg", rimg6)

rimg7 = app.draw_on(img7, faces7)
cv2.imwrite("./nohongcheol7_output.jpg", rimg7)

# similarity 계산
feat1 = np.array(faces1[0].normed_embedding, dtype=np.float32)
feat2 = np.array(faces2[0].normed_embedding, dtype=np.float32)
feat3 = np.array(faces3[0].normed_embedding, dtype=np.float32)
# feat5 = np.array(faces5[0].normed_embedding, dtype=np.float32)
# feat6 = np.array(faces6[0].normed_embedding, dtype=np.float32)
feat7 = np.array(faces7[0].normed_embedding, dtype=np.float32)

sim = np.dot(feat1, feat7.T)
print(sim)