To clip a video using ffmepg: ffmpeg -ss 00:13:10 -t 5 -i Uvic_Bishop_semifinal.mp4 -c:v libx264 -crf 18 -c:a aac clip_1.mp4

## Activate to run SAM3:
1. Activate python environment: source ~/sam3-env/bin/activate
2. Go to the project directory: cd /media/weiweii/Sep/sportsVisCreator

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


### Donuts: Run codes in batch - no bg
*Clip 1*
python scripts/tennis_donuts.py tennis/clip_1.mp4 tennis/rendered_clips/NoBG/clip_1_p1.mp4 --outline 0  --court "549,247;1245,243;1603,899;190,918" --values 0.1509889778  0.8490110222 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_1.mp4 tennis/rendered_clips/NoBG/clip_1_p2.mp4 --outline 0  --court "549,247;1245,243;1603,899;190,918" --values 0.1828476521  0.700739846 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_1.mp4 tennis/rendered_clips/NoBG/clip_1_p3.mp4 --outline 0  --court "549,247;1245,243;1603,899;190,918" --values 0.2215008304  0.578438774 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_1.mp4 tennis/rendered_clips/NoBG/clip_1_p4.mp4 --outline 0  --court "549,247;1245,243;1603,899;190,918" --values 0.2215008304  0.4774271478 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_1.mp4 tennis/rendered_clips/NoBG/clip_1_p5.mp4 --outline 0  --court "549,247;1245,243;1603,899;190,918" --values 0.2684584025  0.4774271478 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_1.mp4 tennis/rendered_clips/NoBG/clip_1_p6.mp4 --outline 0  --court "549,247;1245,243;1603,899;190,918" --values 0.2684584025  0.3940812321 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_1.mp4 tennis/rendered_clips/NoBG/clip_1_p7.mp4 --outline 0  --court "549,247;1245,243;1603,899;190,918" --values 0.3252302582  0.3940812321 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_1.mp4 tennis/rendered_clips/NoBG/clip_1_p8.mp4 --outline 0  --court "549,247;1245,243;1603,899;190,918" --values 0  1  --donuts-only

*Clip 2*
python scripts/tennis_donuts.py tennis/clip_2.mp4 tennis/rendered_clips/NoBG/clip_2_p1.mp4 --outline 0  --court "570,242;1265,238;1625,897;210,910" --values 0.1509889778  0.8490110222 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_2.mp4 tennis/rendered_clips/NoBG/clip_2_p2.mp4 --outline 0  --court "570,242;1265,238;1625,897;210,910" --values 0.1828476521  0.700739846 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_2.mp4 tennis/rendered_clips/NoBG/clip_2_p3.mp4 --outline 0  --court "570,242;1265,238;1625,897;210,910" --values 0.2215008304  0.578438774 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_2.mp4 tennis/rendered_clips/NoBG/clip_2_p4.mp4 --outline 0  --court "570,242;1265,238;1625,897;210,910" --values 0.2215008304  0.4774271478 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_2.mp4 tennis/rendered_clips/NoBG/clip_2_p5.mp4 --outline 0  --court "570,242;1265,238;1625,897;210,910" --values 0.2684584025  0.4774271478 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_2.mp4 tennis/rendered_clips/NoBG/clip_2_p6.mp4 --outline 0  --court "570,242;1265,238;1625,897;210,910" --values 0.2684584025  0.3940812321 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_2.mp4 tennis/rendered_clips/NoBG/clip_2_p7.mp4 --outline 0  --court "570,242;1265,238;1625,897;210,910" --values 0.3252302582  0.3940812321 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_2.mp4 tennis/rendered_clips/NoBG/clip_2_p8.mp4 --outline 0  --court "570,242;1265,238;1625,897;210,910" --values 0  1  --donuts-only

*Clip 3*
python scripts/tennis_donuts.py tennis/clip_3.mp4 tennis/rendered_clips/NoBG/clip_3_p1.mp4 --outline 0  --court "683,277;1273,276;1573,836;384,835" --values 0.1509889778  0.8490110222 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_3.mp4 tennis/rendered_clips/NoBG/clip_3_p2.mp4 --outline 0  --court "683,277;1273,276;1573,836;384,835" --values 0.1828476521  0.700739846 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_3.mp4 tennis/rendered_clips/NoBG/clip_3_p3.mp4 --outline 0  --court "683,277;1273,276;1573,836;384,835" --values 0.2215008304  0.578438774 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_3.mp4 tennis/rendered_clips/NoBG/clip_3_p4.mp4 --outline 0  --court "683,277;1273,276;1573,836;384,835" --values 0.2215008304  0.4774271478 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_3.mp4 tennis/rendered_clips/NoBG/clip_3_p5.mp4 --outline 0  --court "683,277;1273,276;1573,836;384,835" --values 0.2684584025  0.4774271478 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_3.mp4 tennis/rendered_clips/NoBG/clip_3_p6.mp4 --outline 0  --court "683,277;1273,276;1573,836;384,835" --values 0.2684584025  0.3940812321 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_3.mp4 tennis/rendered_clips/NoBG/clip_3_p7.mp4 --outline 0  --court "683,277;1273,276;1573,836;384,835" --values 0.3252302582  0.3940812321 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_3.mp4 tennis/rendered_clips/NoBG/clip_3_p8.mp4 --outline 0  --court "683,277;1273,276;1573,836;384,835" --values 0  1  --donuts-only

*Clip 4*
python scripts/tennis_donuts.py tennis/clip_4.mp4 tennis/rendered_clips/NoBG/clip_4_p1.mp4 --outline 0  --court "632,277;1261,275;1586,874;310,879" --values 0.1509889778  0.8490110222 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_4.mp4 tennis/rendered_clips/NoBG/clip_4_p2.mp4 --outline 0  --court "632,277;1261,275;1586,874;310,879" --values 0.1828476521  0.700739846 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_4.mp4 tennis/rendered_clips/NoBG/clip_4_p3.mp4 --outline 0  --court "632,277;1261,275;1586,874;310,879" --values 0.2215008304  0.578438774 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_4.mp4 tennis/rendered_clips/NoBG/clip_4_p4.mp4 --outline 0  --court "632,277;1261,275;1586,874;310,879" --values 0.2215008304  0.4774271478 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_4.mp4 tennis/rendered_clips/NoBG/clip_4_p5.mp4 --outline 0  --court "632,277;1261,275;1586,874;310,879" --values 0.2684584025  0.4774271478 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_4.mp4 tennis/rendered_clips/NoBG/clip_4_p6.mp4 --outline 0  --court "632,277;1261,275;1586,874;310,879" --values 0.2684584025  0.3940812321 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_4.mp4 tennis/rendered_clips/NoBG/clip_4_p7.mp4 --outline 0  --court "632,277;1261,275;1586,874;310,879" --values 0.3252302582  0.3940812321 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_4.mp4 tennis/rendered_clips/NoBG/clip_4_p8.mp4 --outline 0  --court "632,277;1261,275;1586,874;310,879" --values 0  1  --donuts-only

*Clip 5*
python scripts/tennis_donuts.py tennis/clip_5.mp4 tennis/rendered_clips/NoBG/clip_5_p1.mp4 --outline 0  --court "626,274;1260,270;1587,873;303,880" --values 0.1509889778  0.8490110222 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_5.mp4 tennis/rendered_clips/NoBG/clip_5_p2.mp4 --outline 0  --court "626,274;1260,270;1587,873;303,880" --values 0.1828476521  0.700739846 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_5.mp4 tennis/rendered_clips/NoBG/clip_5_p3.mp4 --outline 0  --court "626,274;1260,270;1587,873;303,880" --values 0.2215008304  0.578438774 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_5.mp4 tennis/rendered_clips/NoBG/clip_5_p4.mp4 --outline 0  --court "626,274;1260,270;1587,873;303,880" --values 0.2215008304  0.4774271478 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_5.mp4 tennis/rendered_clips/NoBG/clip_5_p5.mp4 --outline 0  --court "626,274;1260,270;1587,873;303,880" --values 0.2684584025  0.4774271478 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_5.mp4 tennis/rendered_clips/NoBG/clip_5_p6.mp4 --outline 0  --court "626,274;1260,270;1587,873;303,880" --values 0.2684584025  0.3940812321 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_5.mp4 tennis/rendered_clips/NoBG/clip_5_p7.mp4 --outline 0  --court "626,274;1260,270;1587,873;303,880" --values 0.3252302582  0.3940812321 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_5.mp4 tennis/rendered_clips/NoBG/clip_5_p8.mp4 --outline 0  --court "626,274;1260,270;1587,873;303,880" --values 0  1 --donuts-only


*Clip 6*
python scripts/tennis_donuts.py tennis/clip_6.mp4 tennis/rendered_clips/NoBG/clip_6_p1.mp4 --outline 0  --court "637,262;1234,258;1538,825;335,832" --values 0.1509889778  0.8490110222 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_6.mp4 tennis/rendered_clips/NoBG/clip_6_p2.mp4 --outline 0  --court "637,262;1234,258;1538,825;335,832" --values 0.1828476521  0.700739846 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_6.mp4 tennis/rendered_clips/NoBG/clip_6_p3.mp4 --outline 0  --court "637,262;1234,258;1538,825;335,832" --values 0.2215008304  0.578438774 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_6.mp4 tennis/rendered_clips/NoBG/clip_6_p4.mp4 --outline 0  --court "637,262;1234,258;1538,825;335,832" --values 0.2215008304  0.4774271478 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_6.mp4 tennis/rendered_clips/NoBG/clip_6_p5.mp4 --outline 0  --court "637,262;1234,258;1538,825;335,832" --values 0.2684584025  0.4774271478 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_6.mp4 tennis/rendered_clips/NoBG/clip_6_p6.mp4 --outline 0  --court "637,262;1234,258;1538,825;335,832" --values 0.2684584025  0.3940812321 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_6.mp4 tennis/rendered_clips/NoBG/clip_6_p7.mp4 --outline 0  --court "637,262;1234,258;1538,825;335,832" --values 0.3252302582  0.3940812321 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_6.mp4 tennis/rendered_clips/NoBG/clip_6_p8.mp4 --outline 0  --court "637,262;1234,258;1538,825;335,832" --values 0  1 --donuts-only

*Clip 7*
python scripts/tennis_donuts.py tennis/clip_7.mp4 tennis/rendered_clips/NoBG/clip_7_p1.mp4 --outline 0  --court "659,273;1311,273;1648,896;326,893" --values 0.1509889778  0.8490110222 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_7.mp4 tennis/rendered_clips/NoBG/clip_7_p2.mp4 --outline 0  --court "659,273;1311,273;1648,896;326,893" --values 0.1828476521  0.700739846 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_7.mp4 tennis/rendered_clips/NoBG/clip_7_p3.mp4 --outline 0  --court "659,273;1311,273;1648,896;326,893" --values 0.2215008304  0.578438774 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_7.mp4 tennis/rendered_clips/NoBG/clip_7_p4.mp4 --outline 0  --court "659,273;1311,273;1648,896;326,893" --values 0.2215008304  0.4774271478 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_7.mp4 tennis/rendered_clips/NoBG/clip_7_p5.mp4 --outline 0  --court "659,273;1311,273;1648,896;326,893" --values 0.2684584025  0.4774271478 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_7.mp4 tennis/rendered_clips/NoBG/clip_7_p6.mp4 --outline 0  --court "659,273;1311,273;1648,896;326,893" --values 0.2684584025  0.3940812321 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_7.mp4 tennis/rendered_clips/NoBG/clip_7_p7.mp4 --outline 0  --court "659,273;1311,273;1648,896;326,893" --values 0.3252302582  0.3940812321 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_7.mp4 tennis/rendered_clips/NoBG/clip_7_p8.mp4 --outline 0  --court "659,273;1311,273;1648,896;326,893" --values 0  1 --donuts-only

*Clip 8*
python scripts/tennis_donuts.py tennis/clip_8.mp4 tennis/rendered_clips/NoBG/clip_8_p1.mp4 --outline 0  --court "664,268;1297,266;1624,872;338,870" --values 0.1509889778  0.8490110222 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_8.mp4 tennis/rendered_clips/NoBG/clip_8_p2.mp4 --outline 0  --court "664,268;1297,266;1624,872;338,870" --values 0.1828476521  0.700739846 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_8.mp4 tennis/rendered_clips/NoBG/clip_8_p3.mp4 --outline 0  --court "664,268;1297,266;1624,872;338,870" --values 0.2215008304  0.578438774 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_8.mp4 tennis/rendered_clips/NoBG/clip_8_p4.mp4 --outline 0  --court "664,268;1297,266;1624,872;338,870" --values 0.2215008304  0.4774271478 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_8.mp4 tennis/rendered_clips/NoBG/clip_8_p5.mp4 --outline 0  --court "664,268;1297,266;1624,872;338,870" --values 0.2684584025  0.4774271478 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_8.mp4 tennis/rendered_clips/NoBG/clip_8_p6.mp4 --outline 0  --court "664,268;1297,266;1624,872;338,870" --values 0.2684584025  0.3940812321 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_8.mp4 tennis/rendered_clips/NoBG/clip_8_p7.mp4 --outline 0  --court "664,268;1297,266;1624,872;338,870" --values 0.3252302582  0.3940812321 --donuts-only;
python scripts/tennis_donuts.py tennis/clip_8.mp4 tennis/rendered_clips/NoBG/clip_8_p8.mp4 --outline 0  --court "664,268;1297,266;1624,872;338,870" --values 0  1 --donuts-only
### Donuts: Run codes in batch - with bg 

*Clip 1*
python scripts/tennis_donuts.py tennis/clip_1.mp4 tennis/rendered_clips/clip_1_p1.mp4 --outline 0  --court "549,247;1245,243;1603,899;190,918" --values 0.1509889778  0.8490110222;
python scripts/tennis_donuts.py tennis/clip_1.mp4 tennis/rendered_clips/clip_1_p2.mp4 --outline 0  --court "549,247;1245,243;1603,899;190,918" --values 0.1828476521  0.700739846;
python scripts/tennis_donuts.py tennis/clip_1.mp4 tennis/rendered_clips/clip_1_p3.mp4 --outline 0  --court "549,247;1245,243;1603,899;190,918" --values 0.2215008304  0.578438774;
python scripts/tennis_donuts.py tennis/clip_1.mp4 tennis/rendered_clips/clip_1_p4.mp4 --outline 0  --court "549,247;1245,243;1603,899;190,918" --values 0.2215008304  0.4774271478;
python scripts/tennis_donuts.py tennis/clip_1.mp4 tennis/rendered_clips/clip_1_p5.mp4 --outline 0  --court "549,247;1245,243;1603,899;190,918" --values 0.2684584025  0.4774271478;
python scripts/tennis_donuts.py tennis/clip_1.mp4 tennis/rendered_clips/clip_1_p6.mp4 --outline 0  --court "549,247;1245,243;1603,899;190,918" --values 0.2684584025  0.3940812321;
python scripts/tennis_donuts.py tennis/clip_1.mp4 tennis/rendered_clips/clip_1_p7.mp4 --outline 0  --court "549,247;1245,243;1603,899;190,918" --values 0.3252302582  0.3940812321;
python scripts/tennis_donuts.py tennis/clip_1.mp4 tennis/rendered_clips/clip_1_p8.mp4 --outline 0  --court "549,247;1245,243;1603,899;190,918" --values 0  1 

*Clip 2*
python scripts/tennis_donuts.py tennis/clip_2.mp4 tennis/rendered_clips/clip_2_p1.mp4 --outline 0  --court "570,242;1265,238;1625,897;210,910" --values 0.1509889778  0.8490110222;
python scripts/tennis_donuts.py tennis/clip_2.mp4 tennis/rendered_clips/clip_2_p2.mp4 --outline 0  --court "570,242;1265,238;1625,897;210,910" --values 0.1828476521  0.700739846;
python scripts/tennis_donuts.py tennis/clip_2.mp4 tennis/rendered_clips/clip_2_p3.mp4 --outline 0  --court "570,242;1265,238;1625,897;210,910" --values 0.2215008304  0.578438774;
python scripts/tennis_donuts.py tennis/clip_2.mp4 tennis/rendered_clips/clip_2_p4.mp4 --outline 0  --court "570,242;1265,238;1625,897;210,910" --values 0.2215008304  0.4774271478;
python scripts/tennis_donuts.py tennis/clip_2.mp4 tennis/rendered_clips/clip_2_p5.mp4 --outline 0  --court "570,242;1265,238;1625,897;210,910" --values 0.2684584025  0.4774271478;
python scripts/tennis_donuts.py tennis/clip_2.mp4 tennis/rendered_clips/clip_2_p6.mp4 --outline 0  --court "570,242;1265,238;1625,897;210,910" --values 0.2684584025  0.3940812321;
python scripts/tennis_donuts.py tennis/clip_2.mp4 tennis/rendered_clips/clip_2_p7.mp4 --outline 0  --court "570,242;1265,238;1625,897;210,910" --values 0.3252302582  0.3940812321;
python scripts/tennis_donuts.py tennis/clip_2.mp4 tennis/rendered_clips/clip_2_p8.mp4 --outline 0  --court "570,242;1265,238;1625,897;210,910" --values 0  1 

*Clip 3*
python scripts/tennis_donuts.py tennis/clip_3.mp4 tennis/rendered_clips/clip_3_p1.mp4 --outline 0  --court "683,277;1273,276;1573,836;384,835" --values 0.1509889778  0.8490110222 ;
python scripts/tennis_donuts.py tennis/clip_3.mp4 tennis/rendered_clips/clip_3_p2.mp4 --outline 0  --court "683,277;1273,276;1573,836;384,835" --values 0.1828476521  0.700739846 ;
python scripts/tennis_donuts.py tennis/clip_3.mp4 tennis/rendered_clips/clip_3_p3.mp4 --outline 0  --court "683,277;1273,276;1573,836;384,835" --values 0.2215008304  0.578438774 ;
python scripts/tennis_donuts.py tennis/clip_3.mp4 tennis/rendered_clips/clip_3_p4.mp4 --outline 0  --court "683,277;1273,276;1573,836;384,835" --values 0.2215008304  0.4774271478 ;
python scripts/tennis_donuts.py tennis/clip_3.mp4 tennis/rendered_clips/clip_3_p5.mp4 --outline 0  --court "683,277;1273,276;1573,836;384,835" --values 0.2684584025  0.4774271478 ;
python scripts/tennis_donuts.py tennis/clip_3.mp4 tennis/rendered_clips/clip_3_p6.mp4 --outline 0  --court "683,277;1273,276;1573,836;384,835" --values 0.2684584025  0.3940812321 ;
python scripts/tennis_donuts.py tennis/clip_3.mp4 tennis/rendered_clips/clip_3_p7.mp4 --outline 0  --court "683,277;1273,276;1573,836;384,835" --values 0.3252302582  0.3940812321 ;
python scripts/tennis_donuts.py tennis/clip_3.mp4 tennis/rendered_clips/clip_3_p8.mp4 --outline 0  --court "683,277;1273,276;1573,836;384,835" --values 0  1  

*Clip 4*
python scripts/tennis_donuts.py tennis/clip_4.mp4 tennis/rendered_clips/clip_4_p1.mp4 --outline 0  --court "632,277;1261,275;1586,874;310,879" --values 0.1509889778  0.8490110222 ;
python scripts/tennis_donuts.py tennis/clip_4.mp4 tennis/rendered_clips/clip_4_p2.mp4 --outline 0  --court "632,277;1261,275;1586,874;310,879" --values 0.1828476521  0.700739846 ;
python scripts/tennis_donuts.py tennis/clip_4.mp4 tennis/rendered_clips/clip_4_p3.mp4 --outline 0  --court "632,277;1261,275;1586,874;310,879" --values 0.2215008304  0.578438774 ;
python scripts/tennis_donuts.py tennis/clip_4.mp4 tennis/rendered_clips/clip_4_p4.mp4 --outline 0  --court "632,277;1261,275;1586,874;310,879" --values 0.2215008304  0.4774271478 ;
python scripts/tennis_donuts.py tennis/clip_4.mp4 tennis/rendered_clips/clip_4_p5.mp4 --outline 0  --court "632,277;1261,275;1586,874;310,879" --values 0.2684584025  0.4774271478 ;
python scripts/tennis_donuts.py tennis/clip_4.mp4 tennis/rendered_clips/clip_4_p6.mp4 --outline 0  --court "632,277;1261,275;1586,874;310,879" --values 0.2684584025  0.3940812321 ;
python scripts/tennis_donuts.py tennis/clip_4.mp4 tennis/rendered_clips/clip_4_p7.mp4 --outline 0  --court "632,277;1261,275;1586,874;310,879" --values 0.3252302582  0.3940812321 ;
python scripts/tennis_donuts.py tennis/clip_4.mp4 tennis/rendered_clips/clip_4_p8.mp4 --outline 0  --court "632,277;1261,275;1586,874;310,879" --values 0  1  

*Clip 5*
python scripts/tennis_donuts.py tennis/clip_5.mp4 tennis/rendered_clips/clip_5_p1.mp4 --outline 0  --court "626,274;1260,270;1587,873;303,880" --values 0.1509889778  0.8490110222;
python scripts/tennis_donuts.py tennis/clip_5.mp4 tennis/rendered_clips/clip_5_p2.mp4 --outline 0  --court "626,274;1260,270;1587,873;303,880" --values 0.1828476521  0.700739846;
python scripts/tennis_donuts.py tennis/clip_5.mp4 tennis/rendered_clips/clip_5_p3.mp4 --outline 0  --court "626,274;1260,270;1587,873;303,880" --values 0.2215008304  0.578438774;
python scripts/tennis_donuts.py tennis/clip_5.mp4 tennis/rendered_clips/clip_5_p4.mp4 --outline 0  --court "626,274;1260,270;1587,873;303,880" --values 0.2215008304  0.4774271478;
python scripts/tennis_donuts.py tennis/clip_5.mp4 tennis/rendered_clips/clip_5_p5.mp4 --outline 0  --court "626,274;1260,270;1587,873;303,880" --values 0.2684584025  0.4774271478;
python scripts/tennis_donuts.py tennis/clip_5.mp4 tennis/rendered_clips/clip_5_p6.mp4 --outline 0  --court "626,274;1260,270;1587,873;303,880" --values 0.2684584025  0.3940812321;
python scripts/tennis_donuts.py tennis/clip_5.mp4 tennis/rendered_clips/clip_5_p7.mp4 --outline 0  --court "626,274;1260,270;1587,873;303,880" --values 0.3252302582  0.3940812321
python scripts/tennis_donuts.py tennis/clip_5.mp4 tennis/rendered_clips/clip_5_p8.mp4 --outline 0  --court "626,274;1260,270;1587,873;303,880" --values 0  1

*Clip 6*
python scripts/tennis_donuts.py tennis/clip_6.mp4 tennis/rendered_clips/clip_6_p1.mp4 --outline 0  --court "637,262;1234,258;1538,825;335,832" --values 0.1509889778  0.8490110222;
python scripts/tennis_donuts.py tennis/clip_6.mp4 tennis/rendered_clips/clip_6_p2.mp4 --outline 0  --court "637,262;1234,258;1538,825;335,832" --values 0.1828476521  0.700739846;
python scripts/tennis_donuts.py tennis/clip_6.mp4 tennis/rendered_clips/clip_6_p3.mp4 --outline 0  --court "637,262;1234,258;1538,825;335,832" --values 0.2215008304  0.578438774;
python scripts/tennis_donuts.py tennis/clip_6.mp4 tennis/rendered_clips/clip_6_p4.mp4 --outline 0  --court "637,262;1234,258;1538,825;335,832" --values 0.2215008304  0.4774271478;
python scripts/tennis_donuts.py tennis/clip_6.mp4 tennis/rendered_clips/clip_6_p5.mp4 --outline 0  --court "637,262;1234,258;1538,825;335,832" --values 0.2684584025  0.4774271478;
python scripts/tennis_donuts.py tennis/clip_6.mp4 tennis/rendered_clips/clip_6_p6.mp4 --outline 0  --court "637,262;1234,258;1538,825;335,832" --values 0.2684584025  0.3940812321;
python scripts/tennis_donuts.py tennis/clip_6.mp4 tennis/rendered_clips/clip_6_p7.mp4 --outline 0  --court "637,262;1234,258;1538,825;335,832" --values 0.3252302582  0.3940812321;
python scripts/tennis_donuts.py tennis/clip_6.mp4 tennis/rendered_clips/clip_6_p8.mp4 --outline 0  --court "637,262;1234,258;1538,825;335,832" --values 0  1

*Clip 7*
python scripts/tennis_donuts.py tennis/clip_7.mp4 tennis/rendered_clips/clip_7_p1.mp4 --outline 0  --court "659,273;1311,273;1648,896;326,893" --values 0.1509889778  0.8490110222;
python scripts/tennis_donuts.py tennis/clip_7.mp4 tennis/rendered_clips/clip_7_p2.mp4 --outline 0  --court "659,273;1311,273;1648,896;326,893" --values 0.1828476521  0.700739846;
python scripts/tennis_donuts.py tennis/clip_7.mp4 tennis/rendered_clips/clip_7_p3.mp4 --outline 0  --court "659,273;1311,273;1648,896;326,893" --values 0.2215008304  0.578438774;
python scripts/tennis_donuts.py tennis/clip_7.mp4 tennis/rendered_clips/clip_7_p4.mp4 --outline 0  --court "659,273;1311,273;1648,896;326,893" --values 0.2215008304  0.4774271478;
python scripts/tennis_donuts.py tennis/clip_7.mp4 tennis/rendered_clips/clip_7_p5.mp4 --outline 0  --court "659,273;1311,273;1648,896;326,893" --values 0.2684584025  0.4774271478;
python scripts/tennis_donuts.py tennis/clip_7.mp4 tennis/rendered_clips/clip_7_p6.mp4 --outline 0  --court "659,273;1311,273;1648,896;326,893" --values 0.2684584025  0.3940812321;
python scripts/tennis_donuts.py tennis/clip_7.mp4 tennis/rendered_clips/clip_7_p7.mp4 --outline 0  --court "659,273;1311,273;1648,896;326,893" --values 0.3252302582  0.3940812321;
python scripts/tennis_donuts.py tennis/clip_7.mp4 tennis/rendered_clips/clip_7_p8.mp4 --outline 0  --court "659,273;1311,273;1648,896;326,893" --values 0  1

*Clip 8*
python scripts/tennis_donuts.py tennis/clip_8.mp4 tennis/rendered_clips/clip_8_p1.mp4 --outline 0  --court "664,268;1297,266;1624,872;338,870" --values 0.1509889778  0.8490110222;
python scripts/tennis_donuts.py tennis/clip_8.mp4 tennis/rendered_clips/clip_8_p2.mp4 --outline 0  --court "664,268;1297,266;1624,872;338,870" --values 0.1828476521  0.700739846;
python scripts/tennis_donuts.py tennis/clip_8.mp4 tennis/rendered_clips/clip_8_p3.mp4 --outline 0  --court "664,268;1297,266;1624,872;338,870" --values 0.2215008304  0.578438774;
python scripts/tennis_donuts.py tennis/clip_8.mp4 tennis/rendered_clips/clip_8_p4.mp4 --outline 0  --court "664,268;1297,266;1624,872;338,870" --values 0.2215008304  0.4774271478;
python scripts/tennis_donuts.py tennis/clip_8.mp4 tennis/rendered_clips/clip_8_p5.mp4 --outline 0  --court "664,268;1297,266;1624,872;338,870" --values 0.2684584025  0.4774271478;
python scripts/tennis_donuts.py tennis/clip_8.mp4 tennis/rendered_clips/clip_8_p6.mp4 --outline 0  --court "664,268;1297,266;1624,872;338,870" --values 0.2684584025  0.3940812321;
python scripts/tennis_donuts.py tennis/clip_8.mp4 tennis/rendered_clips/clip_8_p7.mp4 --outline 0  --court "664,268;1297,266;1624,872;338,870" --values 0.3252302582  0.3940812321;
python scripts/tennis_donuts.py tennis/clip_8.mp4 tennis/rendered_clips/clip_8_p8.mp4 --outline 0  --court "664,268;1297,266;1624,872;338,870" --values 0  1


### Bars
*clip_1*
python scripts/tennis_bars.py --perimeter-at center tennis/clip_1.mp4 tennis/rendered_clips_bar/clip_1_p1_bar.mp4 --court "549,247;1245,243;1603,899;190,918" --values 0.1509889778  0.8490110222;
python scripts/tennis_bars.py --perimeter-at center tennis/clip_1.mp4 tennis/rendered_clips_bar/clip_1_p2_bar.mp4 --court "549,247;1245,243;1603,899;190,918" --values 0.1828476521  0.700739846;
python scripts/tennis_bars.py --perimeter-at center tennis/clip_1.mp4 tennis/rendered_clips_bar/clip_1_p3_bar.mp4 --court "549,247;1245,243;1603,899;190,918" --values 0.2215008304  0.578438774;
python scripts/tennis_bars.py --perimeter-at center tennis/clip_1.mp4 tennis/rendered_clips_bar/clip_1_p4_bar.mp4 --court "549,247;1245,243;1603,899;190,918" --values 0.2215008304  0.4774271478;
python scripts/tennis_bars.py --perimeter-at center tennis/clip_1.mp4 tennis/rendered_clips_bar/clip_1_p5_bar.mp4 --court "549,247;1245,243;1603,899;190,918" --values 0.2684584025  0.4774271478;
python scripts/tennis_bars.py --perimeter-at center tennis/clip_1.mp4 tennis/rendered_clips_bar/clip_1_p6_bar.mp4 --court "549,247;1245,243;1603,899;190,918" --values 0.2684584025  0.3940812321;
python scripts/tennis_bars.py --perimeter-at center tennis/clip_1.mp4 tennis/rendered_clips_bar/clip_1_p7_bar.mp4 --court "549,247;1245,243;1603,899;190,918" --values 0.3252302582  0.3940812321
python scripts/tennis_bars.py --perimeter-at center tennis/clip_1.mp4 tennis/rendered_clips_bar/clip_1_p8_bar.mp4 --court "549,247;1245,243;1603,899;190,918" --values 0  0.8490110222

*clip_2*


python scripts/tennis_bars.py --perimeter-at center tennis/clip_5.mp4 tennis/rendered_clips_bar/clip_5_p1_bar.mp4 --court "626,274;1260,270;1587,873;303,880" --values 0.1509889778  0.8490110222;
python scripts/tennis_bars.py --perimeter-at center tennis/clip_5.mp4 tennis/rendered_clips_bar/clip_5_p2_bar.mp4 --court "626,274;1260,270;1587,873;303,880" --values 0.1828476521  0.700739846;
python scripts/tennis_bars.py --perimeter-at center tennis/clip_5.mp4 tennis/rendered_clips_bar/clip_5_p3_bar.mp4 --court "626,274;1260,270;1587,873;303,880" --values 0.2215008304  0.578438774;
python scripts/tennis_bars.py --perimeter-at center tennis/clip_5.mp4 tennis/rendered_clips_bar/clip_5_p4_bar.mp4 --court "626,274;1260,270;1587,873;303,880" --values 0.2215008304  0.4774271478;
python scripts/tennis_bars.py --perimeter-at center tennis/clip_5.mp4 tennis/rendered_clips_bar/clip_5_p5_bar.mp4 --court "626,274;1260,270;1587,873;303,880" --values 0.2684584025  0.4774271478;
python scripts/tennis_bars.py --perimeter-at center tennis/clip_5.mp4 tennis/rendered_clips_bar/clip_5_p6_bar.mp4 --court "626,274;1260,270;1587,873;303,880" --values 0.2684584025  0.3940812321;
python scripts/tennis_bars.py --perimeter-at center tennis/clip_5.mp4 tennis/rendered_clips_bar/clip_5_p7_bar.mp4 --court "626,274;1260,270;1587,873;303,880" --values 0.3252302582  0.3940812321
python scripts/tennis_bars.py --perimeter-at center tennis/clip_5.mp4 tennis/rendered_clips_bar/clip_5_p8_bar.mp4 --court "626,274;1260,270;1587,873;303,880" --values 0  0.8490110222


## Basketball

clips info:

clip_7: 01:38:55

Info of Basketball cout (pass --court to skip clicking):
clip_1 (left_paint): 595,502;1227,556;1070,731;360,662
clip_2 (left_paint): 606,586;1235,641;1080,817;370,746
clip_3 (right_paint): 725,590;1322,541;1550,691;882,755
clip_4 (right_paint): 846,634;1508,582;1766,754;1015,819
clip_5 (left_paint): 720,499;1313,555;1166,717;504,645
clip_6 (left_paint): 643,578;1264,631;1113,805;409,733
clip_7 (right_paint): 722,576;1317,524;1548,675;876,740
clip_8 (left_paint): 655,476;1250,529;1104,694;434,624

To run the pipeline for basketball:
1. Run code for the first time: python scripts/basketball_donuts_frames.py basketball/clip_1.mp4 basketball/rendered_clips/clip_1_p1.mp4 --court-preset left-paint
2. Otherwise: python scripts/basketball_donuts_frames.py basketball/clip_1.mp4 basketball/rendered_clips/clip_1_p1.mp4 --court-preset left-paint --court "595,502;1227,556;1070,731;360,662" --targets 5 6 --values 0.1509889778  0.8490110222

clip_1:
python scripts/basketball_donuts_frames.py basketball/clip_1.mp4 basketball/rendered_clips/clip_1_p1.mp4 --court-preset left-paint --court "595,502;1227,556;1070,731;360,662" --targets 5 6 --values 0.1509889778  0.8490110222;

clip_2:
python scripts/basketball_donuts_frames.py basketball/clip_2.mp4 basketball/rendered_clips/clip_2_test.mp4 --court-preset left-paint --court "606,586;1235,641;1080,817;370,746" --targets-csv basketball/clip_2.csv --values 0.1509889778  0.8490110222;

clip_3:
python scripts/basketball_donuts_frames.py basketball/clip_3.mp4 basketball/rendered_clips/clip_3_test.mp4 --court-preset right-paint --court "725,590;1322,541;1550,691;882,755" --targets 10 5 --values 0.1509889778  0.8490110222

clip_4:
python scripts/basketball_donuts_frames.py basketball/clip_4.mp4 basketball/rendered_clips/clip_4_test.mp4 --court-preset right-paint --court "846,634;1508,582;1766,754;1015,819" --targets 9 6 --values 0.1509889778  0.8490110222;

clip_5:
python scripts/basketball_donuts_frames.py basketball/clip_5.mp4 basketball/rendered_clips/clip_5_test.mp4 --court-preset left-paint --court "720,499;1313,555;1166,717;504,645" --targets 4 5 --values 0.1509889778  0.8490110222

clip_6:
python scripts/basketball_donuts_frames.py basketball/clip_6.mp4 basketball/rendered_clips/clip_6_test.mp4 --court-preset left-paint --court "643,578;1264,631;1113,805;409,733" --targets 4 5 --values 0.1509889778  0.8490110222

clip_7:
python scripts/basketball_donuts_frames.py basketball/clip_7.mp4 basketball/rendered_clips/clip_7_test.mp4 --court-preset right-paint --court "722,576;1317,524;1548,675;876,740" --targets 5 7 --values 0.1509889778  0.8490110222; 

clip_8
python scripts/basketball_donuts_frames.py basketball/clip_8.mp4 basketball/rendered_clips/clip_8_test.mp4 --court-preset left-paint --court "655,476;1250,529;1104,694;434,624" --targets 6 10 --values 0.1509889778  0.8490110222

*Run codes in batch*
clip_3:

python scripts/basketball_donuts_frames.py basketball/clip_3.mp4 basketball/rendered_clips/clip_3_p1.mp4 --court-preset right-paint --court "725,590;1322,541;1550,691;882,755" --targets 10 5 --values 0.1509889778  0.8490110222;
python scripts/basketball_donuts_frames.py basketball/clip_3.mp4 basketball/rendered_clips/clip_3_p2.mp4 --court-preset right-paint --court "725,590;1322,541;1550,691;882,755" --targets 10 5 --values 0.1828476521  0.700739846;
python scripts/basketball_donuts_frames.py basketball/clip_3.mp4 basketball/rendered_clips/clip_3_p3.mp4 --court-preset right-paint --court "725,590;1322,541;1550,691;882,755" --targets 10 5 --values 0.2215008304  0.578438774;
python scripts/basketball_donuts_frames.py basketball/clip_3.mp4 basketball/rendered_clips/clip_3_p4.mp4 --court-preset right-paint --court "725,590;1322,541;1550,691;882,755" --targets 10 5 --values 0.2215008304  0.4774271478;
python scripts/basketball_donuts_frames.py basketball/clip_3.mp4 basketball/rendered_clips/clip_3_p5.mp4 --court-preset right-paint --court "725,590;1322,541;1550,691;882,755" --targets 10 5 --values 0.2684584025  0.4774271478;
python scripts/basketball_donuts_frames.py basketball/clip_3.mp4 basketball/rendered_clips/clip_3_p6.mp4 --court-preset right-paint --court "725,590;1322,541;1550,691;882,755" --targets 10 5 --values 0.2684584025  0.3940812321;
python scripts/basketball_donuts_frames.py basketball/clip_3.mp4 basketball/rendered_clips/clip_3_p7.mp4 --court-preset right-paint --court "725,590;1322,541;1550,691;882,755" --targets 10 5 --values 0.3252302582  0.3940812321;
python scripts/basketball_donuts_frames.py basketball/clip_3.mp4 basketball/rendered_clips/clip_3_p8.mp4 --court-preset right-paint --court "725,590;1322,541;1550,691;882,755" --targets 10 5 --values 0  0.8490110222


clip_4:

python scripts/basketball_donuts_frames.py basketball/clip_4.mp4 basketball/rendered_clips/clip_4_p1.mp4 --court-preset right-paint --court "846,634;1508,582;1766,754;1015,819" --targets 9 6 --values 0.1509889778  0.8490110222;
python scripts/basketball_donuts_frames.py basketball/clip_4.mp4 basketball/rendered_clips/clip_4_p2.mp4 --court-preset right-paint --court "846,634;1508,582;1766,754;1015,819" --targets 9 6 --values 0.1828476521  0.700739846;
python scripts/basketball_donuts_frames.py basketball/clip_4.mp4 basketball/rendered_clips/clip_4_p3.mp4 --court-preset right-paint --court "846,634;1508,582;1766,754;1015,819" --targets 9 6 --values 0.2215008304  0.578438774;
python scripts/basketball_donuts_frames.py basketball/clip_4.mp4 basketball/rendered_clips/clip_4_p4.mp4 --court-preset right-paint --court "846,634;1508,582;1766,754;1015,819" --targets 9 6 --values 0.2215008304  0.4774271478;
python scripts/basketball_donuts_frames.py basketball/clip_4.mp4 basketball/rendered_clips/clip_4_p5.mp4 --court-preset right-paint --court "846,634;1508,582;1766,754;1015,819" --targets 9 6 --values 0.2684584025  0.4774271478;
python scripts/basketball_donuts_frames.py basketball/clip_4.mp4 basketball/rendered_clips/clip_4_p6.mp4 --court-preset right-paint --court "846,634;1508,582;1766,754;1015,819" --targets 9 6 --values 0.2684584025  0.3940812321;
python scripts/basketball_donuts_frames.py basketball/clip_4.mp4 basketball/rendered_clips/clip_4_p7.mp4 --court-preset right-paint --court "846,634;1508,582;1766,754;1015,819" --targets 9 6 --values 0.3252302582  0.3940812321;
python scripts/basketball_donuts_frames.py basketball/clip_4.mp4 basketball/rendered_clips/clip_4_p8.mp4 --court-preset right-paint --court "846,634;1508,582;1766,754;1015,819" --targets 9 6 --values 0  0.8490110222

clip_5:
python scripts/basketball_donuts_frames.py basketball/clip_5.mp4 basketball/rendered_clips/clip_5_p1.mp4 --court-preset left-paint --court "720,499;1313,555;1166,717;504,645" --targets 4 5 --values 0.1509889778  0.8490110222;
python scripts/basketball_donuts_frames.py basketball/clip_5.mp4 basketball/rendered_clips/clip_5_p2.mp4 --court-preset left-paint --court "720,499;1313,555;1166,717;504,645" --targets 4 5 --values 0.1828476521  0.700739846;
python scripts/basketball_donuts_frames.py basketball/clip_5.mp4 basketball/rendered_clips/clip_5_p3.mp4 --court-preset left-paint --court "720,499;1313,555;1166,717;504,645" --targets 4 5 --values 0.2215008304  0.578438774;
python scripts/basketball_donuts_frames.py basketball/clip_5.mp4 basketball/rendered_clips/clip_5_p4.mp4 --court-preset left-paint --court "720,499;1313,555;1166,717;504,645" --targets 4 5 --values 0.2215008304  0.4774271478;
python scripts/basketball_donuts_frames.py basketball/clip_5.mp4 basketball/rendered_clips/clip_5_p5.mp4 --court-preset left-paint --court "720,499;1313,555;1166,717;504,645" --targets 4 5 --values 0.2684584025  0.4774271478;
python scripts/basketball_donuts_frames.py basketball/clip_5.mp4 basketball/rendered_clips/clip_5_p6.mp4 --court-preset left-paint --court "720,499;1313,555;1166,717;504,645" --targets 4 5 --values 0.2684584025  0.3940812321;
python scripts/basketball_donuts_frames.py basketball/clip_5.mp4 basketball/rendered_clips/clip_5_p7.mp4 --court-preset left-paint --court "720,499;1313,555;1166,717;504,645" --targets 4 5 --values 0.3252302582  0.3940812321;
python scripts/basketball_donuts_frames.py basketball/clip_5.mp4 basketball/rendered_clips/clip_5_p8.mp4 --court-preset left-paint --court "720,499;1313,555;1166,717;504,645" --targets 4 5 --values 0  0.8490110222


clip_8:

python scripts/basketball_donuts_frames.py basketball/clip_8.mp4 basketball/rendered_clips/clip_8_p1.mp4 --court-preset left-paint --court "655,476;1250,529;1104,694;434,624" --targets 6 10 --values 0.1509889778  0.8490110222;
python scripts/basketball_donuts_frames.py basketball/clip_8.mp4 basketball/rendered_clips/clip_8_p2.mp4 --court-preset left-paint --court "655,476;1250,529;1104,694;434,624" --targets 6 10 --values 0.1828476521  0.700739846;
python scripts/basketball_donuts_frames.py basketball/clip_8.mp4 basketball/rendered_clips/clip_8_p3.mp4 --court-preset left-paint --court "655,476;1250,529;1104,694;434,624" --targets 6 10 --values 0.2215008304  0.578438774;
python scripts/basketball_donuts_frames.py basketball/clip_8.mp4 basketball/rendered_clips/clip_8_p4.mp4 --court-preset left-paint --court "655,476;1250,529;1104,694;434,624" --targets 6 10 --values 0.2215008304  0.4774271478;
python scripts/basketball_donuts_frames.py basketball/clip_8.mp4 basketball/rendered_clips/clip_8_p5.mp4 --court-preset left-paint --court "655,476;1250,529;1104,694;434,624" --targets 6 10 --values 0.2684584025  0.4774271478;
python scripts/basketball_donuts_frames.py basketball/clip_8.mp4 basketball/rendered_clips/clip_8_p6.mp4 --court-preset left-paint --court "655,476;1250,529;1104,694;434,624" --targets 6 10 --values 0.2684584025  0.3940812321;
python scripts/basketball_donuts_frames.py basketball/clip_8.mp4 basketball/rendered_clips/clip_8_p7.mp4 --court-preset left-paint --court "655,476;1250,529;1104,694;434,624" --targets 6 10 --values 0.3252302582  0.3940812321;
python scripts/basketball_donuts_frames.py basketball/clip_8.mp4 basketball/rendered_clips/clip_8_p8.mp4 --court-preset left-paint --court "655,476;1250,529;1104,694;434,624" --targets 6 10 --values 0  0.8490110222