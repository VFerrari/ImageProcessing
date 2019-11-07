# Victor Ferreira Ferrari, RA 187890
# MC920 - Introduction to Image Digital Processing
# Project 5 - Principal Component Analysis
# Last modified: 07/11/2019

from cv2 import imwrite, imread
from cv2 import imshow, waitKey, destroyAllWindows, IMREAD_COLOR
from os.path import join
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
    
    # Evaluate result
    metrics = eval_compression(img, out)
    print("Compression factor:", metrics['rho'])
    print("Compression RMSE:", metrics['rmse'])
    
    show_image(out, name='Compressed')
    
# Compressing an image via principal component analysis.
# Calculated using SVD.
def pca_compression(img, n_comp):
    
    # Declaring matrices.
    shape = img.shape
    k = n_comp
    U = np.zeros((shape[0],shape[0],shape[2]))
    S = np.zeros(shape)
    V = np.zeros((shape[1],shape[1],shape[2]))
    
    # Divide in 3 channels and apply SVD to each one.
    for i in range(3):
        U[:,:,i], S[:,:,i], V[:,:,i] = np.linalg.svd(img[:,:,i].astype(float))
    
    # Consider k components and combine channels.
    out = np.zeros(shape)
    Ug,Sg,Vg = np.zeros(U.shape),np.zeros(S.shape),np.zeros(V.shape)
    for i in range(3):
        Ug[:,:k,i] = U[:,:k,i]
        Sg[:k,:k,i] = S[:k,:k,i]
        Vg[:k,:,i] = V[:k,:,i]
        out[:,:,i] = np.matmul(np.matmul(Ug[:,:,i],Sg[:,:,i]),Vg[:,:,i])
        
    return out

# Evaluates compression by RMSE and compression factor.
def eval_compression(f, g):
    rho = g.nbytes/f.nbytes
    diff = (f-g)**2
    rmse = np.sqrt(diff.sum()/f.shape)
    
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
