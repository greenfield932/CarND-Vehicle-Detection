python train.py
OUT=$?
if [ $OUT -eq 0 ];then
    python pipeline.py project_video.mp4
fi
#python pipeline.py test_video.mp4
#python pipeline.py project_video.mp4 examples/output_project_video.avi
#python pipeline.py project_video.mp4 
#python pipeline.py challenge_video.mp4
#python pipeline.py challenge_video.mp4 out_challenge_video.avi

#python pipeline.py harder_challenge_video.mp4
#python pipeline.py harder_challenge_video.mp4 examples/out_harder_challenge_video.avi
#python pipeline.py Basler1.avi
#python pipeline.py ~/Downloads/lines.mp4 video1.avi

