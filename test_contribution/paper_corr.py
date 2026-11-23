# Portage NUMPY du code officiel des auteurs (meme algo : images integrales + FFT).
# Permet une comparaison a backend egal (numpy vs numpy) avec notre methode.
import numpy as np

EPS = 1e-17

def integral_image(u):
    assert u.ndim == 2
    if np.iscomplexobj(u):
        v = np.zeros((u.shape[0]+1, u.shape[1]+1), dtype=np.complex128)
    else:
        v = np.zeros((u.shape[0]+1, u.shape[1]+1), dtype=np.float64)
    v[1:,1:] = u
    v = np.cumsum(v, axis=0)
    v = np.cumsum(v, axis=1)
    return v

class CorrHelperNP(object):
    def __init__(self, height, width, fourier_expanded, integral_fourier,
                 integral_fourier_squared, window_ratio, max_distance):
        self.height=height; self.width=width
        self.integral_fourier=integral_fourier
        self.integral_fourier_squared=integral_fourier_squared
        self.patch_height=round(window_ratio*height)
        self.patch_width=round(window_ratio*width)
        offs=[]
        for r in np.arange(0, height-self.patch_height+1, self.patch_height):
            for c in np.arange(0, width-self.patch_width+1, self.patch_width):
                offs.append((r%height, c%width))
        offs=np.array(offs, dtype=np.int64)
        self.row_starts=offs[:,0]; self.col_starts=offs[:,1]
        ph,pw=self.patch_height,self.patch_width
        self.patch_sums=(self.integral_fourier[self.row_starts+ph, self.col_starts+pw]
                        +self.integral_fourier[self.row_starts, self.col_starts]
                        -self.integral_fourier[self.row_starts+ph, self.col_starts]
                        -self.integral_fourier[self.row_starts, self.col_starts+pw])
        self.patch_square_sums=(self.integral_fourier_squared[self.row_starts+ph, self.col_starts+pw]
                        +self.integral_fourier_squared[self.row_starts, self.col_starts]
                        -self.integral_fourier_squared[self.row_starts+ph, self.col_starts]
                        -self.integral_fourier_squared[self.row_starts, self.col_starts+pw])
        self.patch_norms=np.sqrt(self.patch_square_sums-self.patch_sums*np.conj(self.patch_sums)/(ph*pw))
        self.num_patches_row=(height-ph)//ph+1
        self.num_patches_col=(width-pw)//pw+1
        bottom=self.num_patches_row*ph; right=self.num_patches_col*pw
        self.fourier_expanded=fourier_expanded[:, 0:right]
        self.fourier_cropped=fourier_expanded[0:bottom, 0:right]
        self.precomp=np.zeros((max_distance+1, self.num_patches_row, self.num_patches_col), dtype=np.complex128)
        for cj in range(self.num_patches_col):
            img_col=self.fourier_expanded[0:self.height, cj*pw:(cj+1)*pw]
            img_col_fft=np.fft.fft(img_col, axis=0, n=self.height)
            for ri in range(self.num_patches_row):
                patch=self.fourier_cropped[ri*ph:(ri+1)*ph, cj*pw:(cj+1)*pw]
                patch_fft=np.fft.fft(patch, axis=0, n=self.height)
                tr=np.exp(2j*np.pi*np.arange(self.height)*ri*ph/self.height)[:,None]
                corr=np.fft.ifft(patch_fft.conj()*(img_col_fft*tr), axis=0, n=self.height).sum(axis=1).conj()
                self.precomp[:, ri, cj]=corr[:max_distance+1]
        self.precomp=self.precomp.reshape(max_distance+1, self.num_patches_row*self.num_patches_col)

    def compute_correlations(self, distance):
        if distance==0:
            return np.ones(len(self.row_starts))
        ph,pw=self.patch_height,self.patch_width
        rs=np.remainder(self.row_starts+distance, self.height)
        cs=np.remainder(self.col_starts, self.width)
        s2=(self.integral_fourier[rs+ph, cs+pw]+self.integral_fourier[rs, cs]
            -self.integral_fourier[rs+ph, cs]-self.integral_fourier[rs, cs+pw])
        sq2=(self.integral_fourier_squared[rs+ph, cs+pw]+self.integral_fourier_squared[rs, cs]
            -self.integral_fourier_squared[rs+ph, cs]-self.integral_fourier_squared[rs, cs+pw])
        n2=np.sqrt(sq2-s2*np.conj(s2)/(ph*pw))
        sp=self.precomp[distance]
        return np.abs((sp-self.patch_sums*np.conj(s2)/(ph*pw))/(self.patch_norms+EPS)/(n2+EPS))

def compute_corr_distances_np(img_ch, window_ratio, max_distance):
    assert img_ch.ndim==2
    h,w=img_ch.shape[:2]
    u_f=np.fft.fftn(img_ch, axes=(0,1), norm="ortho")
    uf=np.tile(u_f,(2,2))
    ufi=integral_image(uf)
    uf2i=integral_image(uf*np.conjugate(uf)).real
    helper=CorrHelperNP(h,w,uf,ufi,uf2i,window_ratio,max_distance)
    out=[helper.compute_correlations(d) for d in range(0,max_distance+1)]
    return np.stack(out)   # (max_distance+1, nb_windows)