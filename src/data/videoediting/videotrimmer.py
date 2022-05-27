from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import *


directory = "../../../OneDrive - ITU/Bachelor/clean_video/Video19/"

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        print(f)

        path = f

        clip = VideoFileClip(path)
        value = clip.end
        outname = f.split("/")[-1].split(".")[0] + ".mp4"
        outpath = directory + "trimmed/" + outname
        print(outpath)
        ffmpeg_extract_subclip(path, 0, value - 1, targetname=outpath)


