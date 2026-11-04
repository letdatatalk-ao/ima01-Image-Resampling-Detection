import numpy as np
import imageio.v3 as iio
from src.image_io import read_image, rgb_to_luminance, load_gray, save_gray_png


def test_luminance_d_un_gris_est_lui_meme():
    """Si R=G=B=c, la luminance vaut c (les coefficients somment a 1)."""
    c=np.random.default_rng(0).uniform(0,255,(16,16))
    rgb=np.stack([c,c,c],axis=2)
    assert np.allclose(rgb_to_luminance(rgb),c)


def test_luminance_coefficients():
    rgb=np.zeros((1,3,3)); rgb[0,0,0]=255; rgb[0,1,1]=255; rgb[0,2,2]=255
    assert np.allclose(rgb_to_luminance(rgb)[0],[0.299*255,0.587*255,0.114*255])


def test_png_8bits_aller_retour(tmp_path):
    img=np.random.default_rng(1).integers(0,256,(20,30)).astype(float)
    p=str(tmp_path/"a.png"); save_gray_png(img,p)
    assert np.array_equal(load_gray(p),img)


def test_tiff_16bits_ramene_sur_0_255(tmp_path):
    img=np.array([[0,65535],[32768,65535]],dtype=np.uint16)
    rgb=np.stack([img]*3,axis=2)
    p=str(tmp_path/"a.tif"); iio.imwrite(p,rgb)
    out=read_image(p)
    assert out.shape==(2,2,3) and out.max()==255.0 and out.min()==0.0


def test_rgba_alpha_supprime(tmp_path):
    rgba=np.zeros((4,4,4),dtype=np.uint8); rgba[...,3]=255; rgba[...,0]=100
    p=str(tmp_path/"a.png"); iio.imwrite(p,rgba)
    assert read_image(p).shape==(4,4,3)


def test_tiff_compresse_lzw_comme_raise(tmp_path):
    """Les TIFF de RAISE sont compresses en LZW : la lecture doit marcher sans imagecodecs."""
    from PIL import Image
    rgb=np.random.default_rng(2).integers(0,256,(12,10,3)).astype(np.uint8)
    p=str(tmp_path/"lzw.tif"); Image.fromarray(rgb).save(p,compression="tiff_lzw")
    assert np.array_equal(read_image(p),rgb.astype(float))