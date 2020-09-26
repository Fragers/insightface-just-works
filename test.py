import os
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from insightface.embedder import InsightfaceEmbedder

model_path = "models/model/model"
embedder = InsightfaceEmbedder(model_path=model_path, epoch_num='0000', image_size=(112, 112))

img_me1 = cv2.imread("test_images/me1.jpg")

emb_me1 = embedder.embed_image(img_me1, return_all=False)


from insightface import mtcnn_detector

detector = mtcnn_detector.MtcnnDetector(model_folder="insightface/mtcnn-model/")

bboxes, points = detector.detect_face(img_me1)

st_points = (int(bboxes[0][0]), int(bboxes[0][1]))
en_points = (int(bboxes[0][2]), int(bboxes[0][3]))
color = (255, 0, 0)
cv2.rectangle(img_me1, st_points, en_points, color, 2)

font = cv2.FONT_HERSHEY_SIMPLEX
# font settings
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2

img_all = cv2.imread("few_people/22207.jpg")
img_all = cv2.resize(img_all, (1000, 1000))
emb_people = embedder.embed_image(img_all, return_all=True)

bboxes_all, points_all = detector.detect_face(img_all)

cv2.rectangle(img_me1, st_points, en_points, color, 2)
found = list()
all_people_dict = dict()
# get all embeddings
for r, d, f in os.walk('my_homies'):
    for image_path in f:
        cur_image = cv2.imread(os.path.join("my_homies", image_path))
        try:
            cur_emb = embedder.embed_image(cur_image, return_all=False)
            all_people_dict[image_path] = cur_emb
            print(str(image_path))
        except:
            continue


for people_index in range(len(bboxes_all) - 1):
    # for alone
    emb_on_photo = emb_people[people_index]
    st_points = (int(bboxes_all[people_index][0]), int(bboxes_all[people_index][1]))
    en_points = (int(bboxes_all[people_index][2]), int(bboxes_all[people_index][3]))
    cv2.rectangle(img_all, st_points, en_points, color, 2)
    flag = False
    #cv2.putText(img_all, str(people_index), st_points, font, fontScale, color, thickness, cv2.LINE_AA)
    for image_path in all_people_dict:
        # if image_path in found:
        #     continue
        cur_emb = all_people_dict[image_path]
        if euclidean(emb_on_photo, cur_emb) <= 1.0:
            cv2.putText(img_all, image_path, st_points, font,
                        fontScale, color, thickness, cv2.LINE_AA)
            found.append(image_path)
            print("found: ", euclidean(emb_on_photo, cur_emb), image_path)
            flag = True
            break
        print(euclidean(emb_on_photo, cur_emb), image_path)
    if not flag:
        cv2.putText(img_all, "unknown", st_points, font,
                    fontScale, color, thickness, cv2.LINE_AA)
cv2.imshow('123', img_all)
cv2.imwrite('examples/example.jpg', img_all)
print(found)
cv2.waitKey(0)
cv2.destroyAllWindows()
