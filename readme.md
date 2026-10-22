To clip a video using ffmepg: ffmpeg -ss 00:13:10 -t 5 -i Uvic_Bishop_semifinal.mp4 -c:v libx264 -crf 18 -c:a aac clip_1.mp4

## Activate to run SAM3:
    a. Activate python environment: source ~/sam3-env/bin/activate
    b. Go to the project directory: cd /media/weiweii/Sep/sportsVisCreator



## Tennis
To run the pipeline for tennis:
1. Run code for the first time: python scripts/tennis_donuts.py tennis/clip_2.mp4 tennis/rendered_clips/clip_2.mp4 --values 0.166 0.937
2. Otherwise: python scripts/tennis_donuts.py tennis/clip_2.mp4 tennis/rendered_clips/clip_2.mp4 --court "683,277;1273,276;1573,836;384,835" --values 0.166 0.937

info of Tennis (pass --court to skip clicking):
clip_1: 549,247;1245,243;1603,899;190,918
clip_2: 570,242;1265,238;1625,897;210,910
clip_3: 683,277;1273,276;1573,836;384,835
clip_4: 632,277;1261,275;1586,874;310,879
clip_5: 626,274;1260,270;1587,873;303,880
clip_6: 637,262;1234,258;1538,825;335,832
clip_7: 659,273;1311,273;1648,896;326,893
clip_8: 664,268;1297,266;1624,872;338,870


*Run codes in batch*
python scripts/tennis_donuts.py tennis/clip_5.mp4 tennis/rendered_clips/clip_5_p1.mp4 --court "626,274;1260,270;1587,873;303,880" --values 0.1509889778  0.8490110222;
python scripts/tennis_donuts.py tennis/clip_5.mp4 tennis/rendered_clips/clip_5_p2.mp4 --court "626,274;1260,270;1587,873;303,880" --values 0.1828476521  0.700739846;
python scripts/tennis_donuts.py tennis/clip_5.mp4 tennis/rendered_clips/clip_5_p3.mp4 --court "626,274;1260,270;1587,873;303,880" --values 0.2215008304  0.578438774;
python scripts/tennis_donuts.py tennis/clip_5.mp4 tennis/rendered_clips/clip_5_p4.mp4 --court "626,274;1260,270;1587,873;303,880" --values 0.2215008304  0.4774271478;
python scripts/tennis_donuts.py tennis/clip_5.mp4 tennis/rendered_clips/clip_5_p5.mp4 --court "626,274;1260,270;1587,873;303,880" --values 0.2684584025  0.4774271478;
python scripts/tennis_donuts.py tennis/clip_5.mp4 tennis/rendered_clips/clip_5_p6.mp4 --court "626,274;1260,270;1587,873;303,880" --values 0.2684584025  0.3940812321;
python scripts/tennis_donuts.py tennis/clip_5.mp4 tennis/rendered_clips/clip_5_p7.mp4 --court "626,274;1260,270;1587,873;303,880" --values 0.3252302582  0.3940812321

## Basketball

Info of Basketball cout (pass --court to skip clicking):
clip_1 (left_paint): 595,502;1227,556;1070,731;360,662
clip_2 (left_paint): 606,586;1235,641;1080,817;370,746
clip_3 (right_paint): 725,590;1322,541;1550,691;882,755
clip_4: 632,277;1261,275;1586,874;310,879
clip_5: 626,274;1260,270;1587,873;303,880
clip_6: 637,262;1234,258;1538,825;335,832
clip_7: 659,273;1311,273;1648,896;326,893
clip_8: 664,268;1297,266;1624,872;338,870

To run the pipeline for tennis:
1. Run code for the first time: python scripts/basketball_donuts_frames.py basketball/clip_1.mp4 basketball/rendered_clips/clip_1_p1.mp4 --court-preset left-paint
2. Otherwise: python scripts/basketball_donuts_frames.py basketball/clip_1.mp4 basketball/rendered_clips/clip_1_p1.mp4 --court-preset left-paint --court "595,502;1227,556;1070,731;360,662" --targets 5 6 --values 0.1509889778  0.8490110222

*Run codes in batch (for clip_1, update targets and court for other clips)*
python scripts/basketball_donuts_frames.py basketball/clip_1.mp4 basketball/rendered_clips/clip_1_p1.mp4 --court-preset left-paint --court "595,502;1227,556;1070,731;360,662" --targets 5 6 --values 0.1509889778  0.8490110222;
python scripts/basketball_donuts_frames.py basketball/clip_1.mp4 basketball/rendered_clips/clip_1_p2.mp4 --court-preset left-paint --court "595,502;1227,556;1070,731;360,662" --targets 5 6 --values 0.1828476521  0.700739846;
python scripts/basketball_donuts_frames.py basketball/clip_1.mp4 basketball/rendered_clips/clip_1_p3.mp4 --court-preset left-paint --court "595,502;1227,556;1070,731;360,662" --targets 5 6 --values 0.2215008304  0.578438774;
python scripts/basketball_donuts_frames.py basketball/clip_1.mp4 basketball/rendered_clips/clip_1_p4.mp4 --court-preset left-paint --court "595,502;1227,556;1070,731;360,662" --targets 5 6 --values 0.2215008304  0.4774271478;
python scripts/basketball_donuts_frames.py basketball/clip_1.mp4 basketball/rendered_clips/clip_1_p5.mp4 --court-preset left-paint --court "595,502;1227,556;1070,731;360,662" --targets 5 6 --values 0.2684584025  0.4774271478;
python scripts/basketball_donuts_frames.py basketball/clip_1.mp4 basketball/rendered_clips/clip_1_p6.mp4 --court-preset left-paint --court "595,502;1227,556;1070,731;360,662" --targets 5 6 --values 0.2684584025  0.3940812321;
python scripts/basketball_donuts_frames.py basketball/clip_1.mp4 basketball/rendered_clips/clip_1_p7.mp4 --court-preset left-paint --court "595,502;1227,556;1070,731;360,662" --targets 5 6 --values 0.3252302582  0.3940812321
