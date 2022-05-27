import json
import math


class padder:

    def __init__(self):
        self.full_skeleton = None

    def open_json(self, path_to_json):
        skeleton_json = open(path_to_json)
        self.full_skeleton = json.load(skeleton_json)

    def pad_video(self, length, video):

        skeleton = video[0]
        labels = video[1]

        rounds = math.ceil(length/len(skeleton))


        padded_video =  []
        skeleton_temp = skeleton
        labels_temp = labels

        for i in range(rounds):
            skeleton_temp.extend(skeleton)
            labels_temp.extend(labels)

        padded_video.append(skeleton_temp[:length])
        padded_video.append(labels_temp[:length])

        return padded_video

    def json_create(self):
        path_to_json = r'/Users/Morten/Library/CloudStorage/OneDrive-SharedLibraries-ITU/Sofus Sebastian Schou Konglevoll - Bachelor/all_skeleton/skeleton_good_padded_videos_v2.json'

        with open(path_to_json, 'w') as outfile:
            json.dump(self.full_skeleton, outfile)

    def run(self, length):


        for key in self.full_skeleton.keys():

            video = self.full_skeleton[key]
            padded_video = self.pad_video(length, video)
            self.full_skeleton[key] = padded_video
        # print(self.full_skeleton['Video1_Video3'])
        self.json_create()








def main():

    path = r'/Users/Morten/Library/CloudStorage/OneDrive-SharedLibraries-ITU/Sofus Sebastian Schou Konglevoll - Bachelor/all_skeleton/skeleton_from_csv_v2.json'
    pad = padder()
    pad.open_json(path)
    pad.run(300)


if __name__ == '__main__':
    main()



