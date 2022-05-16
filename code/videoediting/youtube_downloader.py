"""Downloades video from youtube given an URL"""

import youtube_dl
def run(URL=None):

    video_url = URL
    video_info = youtube_dl.YoutubeDL().extract_info(
        url = video_url,download=False
    )
    filename = f"{video_info['title']}.mp4"
    options={
        'format': "134", #'bestvideo/best'
        'keepvideo':True,
        'outtmpl':filename,
    }

    with youtube_dl.YoutubeDL(options) as ydl:
        ydl.download([video_info['webpage_url']])

    print("Download complete... {}".format(filename))
#    print([x for x in [video_info]])

if __name__=='__main__':
    data = open("url.txt", "r").readlines()
    for x in data:


        print("Working on url:", x ,"\n")
        run(x)
        print("Done with url:", x ,"\n")

        if data == "end":
            break

        


