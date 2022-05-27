
import os
import shutil

# os.rename("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
# os.replace("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
# shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")



folders = ["Video5", "Video4", "Video14", "Video12", "Video15", "Video16", "Video17", "Video18", "Video19"]


directory = "../../../OneDrive - ITU/Bachelor/clean_video/"

for folder in os.listdir(directory):
    if folder in folders:
        f = os.path.join(directory, folder)
        # checking if it is a file
        # print("folder:",f.split("/")[-1])
        for subfolder in os.listdir(f):

            if subfolder == "trimmed":
                f = os.path.join(f, subfolder)
                for file in os.listdir(f)

                    try:
                        if file.split(".")

                    file_path = os.path.join(f, file)



                print(len(os.listdir(f)))

                # for videoName in os.listdir(f):
                #
                #     v = videoName.split(".")[0]
                #     new_f = os.path.join(f, v)
                #     try:
                #         os.mkdir(new_f)
                #     except FileExistsError:
                #         pass
                #
                #     shutil.copyfile(src, dst)
                #
                #
                # break
                #
                #





    #
    # if os.path.isfile(f):
    #     print(f)
