import argparse
import matplotlib.pyplot as plt
import glob
import os
from model_files import *

for path in glob.glob("../VideoToFrame/output/*"):
    colorizer_siggraph17 = siggraph17(pretrained=True).eval()
    img = load_img(path)
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))

    img_bw = postprocess_tens(tens_l_orig, torch.cat(
        (0*tens_l_orig, 0*tens_l_orig), dim=1))
    out_img_siggraph17 = postprocess_tens(
        tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

    plt.imsave('%s%s_siggraph.png' %
               ("../Remastering/model/output/", path[23:-4]), out_img_siggraph17)
