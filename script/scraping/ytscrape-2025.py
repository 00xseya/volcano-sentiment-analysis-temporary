from ytscrape import YouTube, SearchFilter

with YouTube(language="id", region="ID") as yt:
    for video in yt.search(
        "erupsi gunung 2025",
        filter=SearchFilter.VIDEOS,
        max_results=20
    ):
        print(video.title, "-", video.url)