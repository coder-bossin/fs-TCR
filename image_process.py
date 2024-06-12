import os
import operator
import cv2


def get_all_files_pth(dir_path: str, paths_list, suffix: str):

	if os.path.isfile(dir_path) and operator.contains(dir_path, suffix):

		paths_list.append(dir_path)

	elif os.path.isdir(dir_path):

		for s in os.listdir(dir_path):

			newDir = os.path.join(dir_path, s)

			get_all_files_pth(newDir, paths_list, suffix)

	return paths_list

def cal_mean_std(images_dir: str):

	paths_list=get_all_files_pth(images_dir,[],'.jpg')
	print(len(paths_list))
	mean_total=0
	std_total=0
	for path in paths_list:
		temp=cv2.imread(path,0)
		(mean, std) = cv2.meanStdDev(temp)

		mean_total=mean_total+mean
		std_total=std_total+std

	mean_total=mean_total/len(paths_list)
	std_total=std_total/len(paths_list)

	mean_std_dict = {'mean': mean_total[0][0]/255, 'std': std_total[0][0]/255}
	return mean_std_dict


def get_file_name_from_pth(fpth: str):

	file_Names = []

	filelist = os.listdir(fpth)

	for file in filelist:
		filename = file.split(".")[0]
		file_Names.append(filename)

	return file_Names

if __name__ == "__main__":
	img_dir = r''
	mean_std_dict = cal_mean_std(img_dir)
	print(mean_std_dict)