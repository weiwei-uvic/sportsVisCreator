To clip a video using ffmepg: ffmpeg -ss 00:13:10 -t 5 -i Uvic_Bishop_semifinal.mp4 -c:v libx264 -crf 18 -c:a aac clip_1.mp4


Pipeline to run SAM3
1. open wsl
2. activate python environment: source ~/sam3-env/bin/activate
cd /media/weiweii/Sep/sportsVisCreator

3. login huggingface: hf auth login; hf auth whoami
4. go to the working directory: cd /mnt/c/Users/reren/Documents/UVic/Mitacs26/Codes
5. run the python code: python add_tennis_donuts.py --input videoClips/tennis/clip_3.mp4
6. enter one seed point inside each player when prompted, then enter the four court corners

The seed points initialize two separate SAM 3 tracks. To reuse the same seed points on a repeated run, pass `--player-points "x,y; x,y"`. The old `--prompt` option is accepted but ignored.

The tennis script uses `yolo11n-pose.pt` ankle keypoints for foot placement. The basketball script uses pose keypoints only to seed SAM across each target player's body; donut positions then come from the bottom of each target's SAM mask. YOLO11 segmentation masks cover other people to provide occlusion. Ultralytics downloads `yolo11n-seg.pt` on the first run; override it with `--segment-model` if needed. Install the runtime in the active environment with `python -m pip install ultralytics`.


Tennis corner points (in order: nearleft; nearright; farright; farleft):
clip_3: 381,839; 1579,840; 1274,275; 683,276
clip_4: 307,882; 1589,876; 1263,274; 631,277
clip_5: 325,840; 1541,831; 1233,261; 631,264
clip_7: 659,273;1313,272;1652,899;322,896 
clip_8: 663,267;1300,266;1628,875;337,873

1560,425;1380,442
360,662;1075,732;1231,557;596,502
1150,599;1290,773
370,747;1081,817;1234,641;608,586
