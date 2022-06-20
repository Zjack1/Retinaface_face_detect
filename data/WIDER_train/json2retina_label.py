import os
import json
import cv2
test_all_json_file_path = "./shake_json"
test_txt_list = open("./train_shake_list.txt", "w")
image_file = "./shake/"


def get_all_files(dir):
    files_ = []
    list = os.listdir(dir)
    for i in range(0, len(list)):
        path = os.path.join(dir, list[i])
        if os.path.isdir(path):
            files_.extend(get_all_files(path))
        if os.path.isfile(path):
            files_.append(path)
    return files_, list


root = os.path.dirname(os.path.realpath(__file__))
all_json_file, all_json_name = get_all_files(test_all_json_file_path)
len_all_json_file = len(all_json_file)

for i in range(len_all_json_file):

        one_img_path = image_file + all_json_name[i][:-5] + ".jpg"  # 判断每个json是否有对应的图片
        img = cv2.imread(one_img_path)
        if img is None:
            continue
        print("loading ", i, " json file")
        one_json_file = all_json_file[i]
        print(one_json_file)
        f = open(one_json_file)
        ls = json.load(f)
        face_box = ls["shapes"][0]["points"]
        face_box_label = [str(int(i)) for item in face_box for i in item]
        if len(face_box_label) != 4:
            print("face_box_label!=4")
            continue
        face_w = int(face_box_label[2]) - int(face_box_label[0])
        face_h = int(face_box_label[3]) - int(face_box_label[1])
        if face_w < 30:
            continue
        test_txt_list.writelines("# " + all_json_name[i][:-5] + ".jpg" + "\n")
        test_txt_list.writelines(str(face_box_label[0]) + ' ' + str(face_box_label[1]) + ' ' + str(face_w) + ' ' + str(face_h) +  # 为了适配原本格式是训练输入
                                " -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 0.27" + "\n")

test_txt_list.close()
