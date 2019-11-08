# Victor Ferreira Ferrari, RA 187890
# MC920 - Introduction to Image Digital Processing
# Project 5 - Principal Component Analysis
# Last modified: 08/11/2019

from cv2 import imwrite, imread
from cv2 import imshow, waitKey, destroyAllWindows, IMREAD_COLOR
from os.path import join, basename, getsize
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description='Uses Principal Component Analysis (PCA) for compression.')
    parser.add_argument('file', help='Name of the file containing the image (PNG)')
    parser.add_argument('comp', help='Amount of components to split.', type=int)
    parser.add_argument('--plot', help='Plot component variance graph.', action='store_true')
    parser.add_argument('--folder', help='Folder to save compressed image (defaults to Outputs/).', default='Outputs')
    args = parser.parse_args()
    
    # Read color image
    img = imread(args.file, IMREAD_COLOR)
    
    # Compress image
    out = pca_compression(img, args.comp)
    
    # Show/Save result
    name = basename(args.file)
    show_image(out.astype(np.uint8), name='Compressed')
    save_image(out, name, args.folder)
    
    # Evaluate result
    metrics = eval_compression(img.astype(np.float64), out, args.file, join(args.folder, name))
    print("Compression factor:", metrics['rho'])
    print("Compression RMSE:", metrics['rmse'])

    
# Compressing an image via principal component analysis.
# Calculated using SVD.
def pca_compression(img, n_comp):
    k = n_comp
    
    # Divide in 3 channels (BGR) and apply SVD to each one.
    USVb = np.linalg.svd(img[:,:,0], full_matrices=False)
    USVg = np.linalg.svd(img[:,:,1], full_matrices=False)
    USVr = np.linalg.svd(img[:,:,2], full_matrices=False)
    
    U = np.dstack((USVb[0],USVg[0],USVr[0]))
    S = np.dstack((USVb[1],USVg[1],USVr[1]))
    V = np.dstack((USVb[2],USVg[2],USVr[2]))
    
    # Consider k components and combine channels.
    out = np.zeros(img.shape)
    
    Ug = U[:,:k,:]
    Sg = S[:k,:k,:]
    Vg = V[:k,:,:]
    
    for i in range(3):
        out[:,:,i] = (Ug[:,:,i]*Sg[:,:,i]) @ Vg[:,:,i]
    
    return out

# Evaluates compression by RMSE and compression factor.
def eval_compression(f, g, f_path, g_path):
    rho = getsize(g_path)/getsize(f_path)
    
    diff = (f-g)**2
    rmse = np.sqrt(diff.sum().sum()/(f.shape[0] * f.shape[1]))
    
    return {'rho':rho, 'rmse':rmse}

# Self-explanatory
def show_image(img, name='Image'):
    imshow(name, img)
    waitKey(0)
    destroyAllWindows()

def save_image(img, name, folder):
    imwrite(join(folder, name), img)
    
if __name__ == "__main__":
    main()
