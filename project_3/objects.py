# Victor Ferreira Ferrari, RA 187890
# MC920 - Introduction to Image Digital Processing
# Project 3 - Properties of Objects in Images
# Last modified: 10/10/2019

from os.path import join, basename
from sys import argv
from skimage import img_as_float
from skimage.measure import label, regionprops
import numpy as np
import matplotlib.pyplot as plt
import cv2

def main():
    
    # Argument parsing
    argc = len(argv)
    if argc < 2:
        print("Please pass an image as argument!")
    filename = argv[1]
    folder = argv[2] if argc > 2 else 'Outputs'
    
    # Read image
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    show_image(img, name='Colored Image')
    
    # Split regions in image, by label and contours.
    label, contours, cont_img = region_split(img)
    show_image(cont_img, name='Contours')
    save_image(cont_img, filename, 'contours')
    
    # Add number to each region in image, at the centroid.
    numbered = number_regions(img, contours)
    show_image(numbered, name='Labels')
    save_image(numbered, filename, 'labeled')
    
    # Extract properties from objects.
    prop = object_properties(label)
    
    # Make an area histogram for the image.
    name = 'histogram_' + basename(filename)
    area_histogram(prop, filename)
    plt.savefig(join(folder,name))
    plt.show()

# Splits regions on an image.
# Image has to have gray objects on a white background.
# Return region contours and labeled image.
def region_split(img):
    
    # Grayscale 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    # Threshold for black background and white objects.
    _, gray = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
    show_image(gray, name='Binary Image')
    
    # Label image
    gray2 = img_as_float(gray)
    labeled = label(gray2)
    
    # Find object contours and show.
    cont, _= cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    gray[:] = 255
    cv2.drawContours(gray, cont, -1, (0,0,255), 1)
    
    return labeled, cont, gray

# Add label number for each region on its centroid.
# Returns numbered image.
def number_regions(img, contours):
    n = len(contours)-1
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.3
    thick = 0
    
    # For each contour, add label to centroid
    for i,cnt in enumerate(contours):
        label = n-i
        text_size = cv2.getTextSize(str(label), font, scale, thick)[0]
        
        # Centroid (corrected for centering text) from moments.
        m = cv2.moments(cnt)
        x = round(m['m10'] / m['m00'])
        y = round(m['m01'] / m['m00'])
        textX = x-(text_size[0]//2)
        textY = y+(text_size[1]//2)
        
        cv2.putText(img, f'{label}', (textX, textY), font, fontScale=scale, color=(0,255,255), thickness=thick, lineType=cv2.LINE_AA)
        
    return img

# Get object properties from labeled image.
# Print some properties.
# Properties: Area, Perimeter, Eccentricity, Solidity
def object_properties(label):
    regions = regionprops(label, coordinates='rc')
    
    print("número de regiões: ", len(regions))
    print()
    
    # Properties for every labeled region.
    for i,reg in enumerate(regions):
        centroid = (int(round(reg.centroid[0])), int(round(reg.centroid[1])))
        print(f'região {i}:', end=' ')
        print(f' área: {reg.area:4d}', end=' ')
        print(f' perímetro: {reg.perimeter:10.6f}', end=' ')
        print(f' excentricidade: {reg.eccentricity:.6f}', end=' ')
        print(f' solidez: {reg.solidity:.6f}', end=' ')
        print(f' centróide: {centroid}')
    print()
    
    return regions
    
# Create an area size histogram.
# Classifies regions in small, medium and large
def area_histogram(properties, filename):
    sizes = np.zeros(3).astype(np.uint8)
    areas = []
    
    for reg in properties:
        size = min(reg.area//1500, 2)
        sizes[size]+=1
        areas.append(reg.area)
    
    print('número de regiões pequenas:', sizes[0])
    print('número de regiões médias:', sizes[1])
    print('número de regiões grandes:', sizes[2])
    
    plt.hist(areas, bins=3, ec='black', color='blue')
    plt.xlabel("Área")
    plt.ylabel("Número de Objetos")
    plt.title(f"Histograma de Áreas dos Objetos de {basename(filename)}")
    plt.xticks(range(0, max(areas)+1000, 500))

# Self-explanatory
def show_image(img, name='Image'):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_image(img, path, prefix, folder='Outputs'):
    name = prefix + '_' + basename(path)
    cv2.imwrite(join(folder, name), img)

if __name__ == "__main__":
    main()
