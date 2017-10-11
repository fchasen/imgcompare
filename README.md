Feature-based image matching sample.
================================

https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_surf_intro/py_surf_intro.html

Getting Started
-------------------------
Install opencv3 with homebrew

`brew install opencv`

Usage
-------------------------
```
  find_obj.py [--feature=<sift|surf|orb|akaze|brisk>[-flann]] [ <image1> <image2> ]

  --feature  - Feature to use. Can be sift, surf, orb or brisk. Append '-flann'
               to feature name to use Flann-based matcher instead bruteforce.
```

Examples
-------------------------

Still Life with Blue Pot (SURF + Flann):

`./match.py images/stilllife_getty.jpg images/stilllife_img2.png` => 88% matched

Still Life with Blue Pot (SIFT):

`./match.py --feature=sift images/stilllife_getty.jpg images/stilllife_img2.png` => 93% matched

Young Italian Woman at a Table (SURF + Flann):

`./match.py images/woman_getty.jpg images/woman_pdf.jpg ` => 96% matched
