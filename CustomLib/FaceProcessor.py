####
## Import libraries
####

import sys
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
#from matplotlib import cm

####
## Global variables
####

DATASET_DIR = "assets/rawdata/"
OUTPUT_DIR = "assets/output/"
IMAGE_COUNT = 3000
N_COMPONENTS = 200

####
## Class definition
####

class ImageProcessor:

    ## Class constructor
    def __init__(self) -> None:
        self.imagefile1 = None
        self.imagefile2 = None
        self.dataset = []
        self.V = None
        self.meanFace = None
        self.min_rows, self.min_cols = sys.maxsize, sys.maxsize
        self.max_rows, self.max_cols = 0, 0

    ## Recenter image
    def recenter(self,
                 image : np.ndarray) -> np.ndarray:
        
        r, c = image.shape
        top, bot, left, right = 0, r, 0, c
        if r > self.min_rows:
            top = r - self.min_rows  
        if c > self.min_cols:
            right = self.min_cols

        return image[top:bot, left:right]

    ## Import an image from the given path
    def import_image(self, file_path : str, is_dataset = True) -> np.ndarray:
        try:
            if is_dataset:
                img = np.fromfile(file_path, dtype=np.uint8).reshape((128, 128))
            else:
                img = Image.open(file_path).convert("L").resize((self.min_rows, self.min_cols))
                img = self.recenter(np.array(img, dtype= np.uint8))
            return img
        except Exception as e:
            print(e)
            pass
    
    ## Import the dataset
    def get_dataset(self, dataset_dir : str):
        original_images = list(map(
            lambda i : self.import_image(dataset_dir + str(i))
            ,np.random.randint(1223, 5222, IMAGE_COUNT)))
        
        # Remove None values
        original_images = [x for x in original_images if x is not None]
        
        print(f"Imported {len(original_images)} images")

        min_rows, min_cols = self.min_rows, self.min_cols
        max_rows, max_cols = self.max_rows, self.max_cols

        for i,image in enumerate(original_images):
            r, c = image.shape[0], image.shape[1]    
            min_rows = min(min_rows, r)
            max_rows = max(max_rows, r)
            min_cols = min(min_cols, c)
            max_cols = max(max_cols, c)

        print("\n==> Least common image size:", min_rows, "x", min_cols, "pixels")

        self.min_rows, self.min_cols = min_rows, min_cols
        self.max_rows, self.max_cols = max_rows, max_cols

        test_img = list(map(lambda img : self.recenter(img), original_images))

        # Create m x d data matrix
        m = len(test_img)
        d = min_rows * min_cols
        self.dataset = np.reshape(test_img, (m, d))
    
    def compute_mean(self):
        self.meanFace = self.dataset.mean(axis=0)

    def compute_eigV(self):
        #A matrix (face set minus average face)
        A = (self.dataset - self.meanFace).T
        M = A.T @ A
        # Computing eigenvectors and eigenvalues of A.T @ A
        mi, V = np.linalg.eig(M)
        # Computing main eigenvectors
        U = A @ V

        # Normalized eigenvectors
        eigVnorm = U / np.linalg.norm(U, axis=0)

        # Sorted eigenvalues
        sorted_mi = np.sort(mi)[::-1]
        sorted_i = np.argsort(mi)[::-1]
        # Sorted eigenvectors by significance from eigenvalues
        self.V = np.take(eigVnorm, sorted_i, axis=1)

    ## Reconstruct image as a linear convination of eigenvectors
    def reconstruct_img(self, 
                       target) -> Image:
        
        # Get eigenvalues by solving V @ U = Target
        values, residuals, rank, singular = np.linalg.lstsq(self.V, target - self.meanFace, rcond=None)
        # Reconstruct image by multiplying egenvetors by eigenvalues
        reconstruct = self.V @ values
        # Image as squared array
        reconstruct = np.reshape(reconstruct + self.meanFace,(128,128))

        # Return result as an image, as well as the residual error
        return Image.fromarray(np.abs(reconstruct).astype(np.uint8)).convert("L"), residuals
    
    ## Reconstruct an image as a linear combination of the eigenvectors and create an animation
    def reconstruct_from_V(self,
                         target,
                         components,
                         name = OUTPUT_DIR + "res.gif",
                         t = 100):
        
        #Arreglo para guardar la animacion
        frames = []
        values, residuals, rank, singular = np.linalg.lstsq(self.V[:,:components],
                                                            target - self.meanFace,
                                                            rcond=None) #Obtener los eigenvalores de la imagen segun los vectores caracteristicos
        # Cre
        for f in range(values.shape[0]):
            # Get eigenvalues by solving V @ U = Target
            reconstruct = self.V[:,:f+1] @ values[:f+1]
            # Image as squared array
            reconstruct = np.reshape(reconstruct + self.meanFace,(128,128))
            # Return result as an image, as well as the residual error
            resIm = Image.fromarray(np.abs(reconstruct).astype(np.uint8)).convert("L")
            # Append image to the list of frames
            frames.append(resIm)

        # Save animation
        frames[0].save(name,
                       duration=t,
                       loop=0,
                       save_all=True, append_images=frames[1:], optimize=False)
        
    def blend_images(self,
                        target1,
                        target2,
                        components,
                        name = OUTPUT_DIR + "res.gif",
                        t = 100):
        values1, residuals1, rank1, singular1 = np.linalg.lstsq(self.V[:,:components],
                                                            target1 - self.meanFace,
                                                            rcond = None)
        
        values2, residuals2, rank2, singular2 = np.linalg.lstsq(self.V[:,:components],
                                                            target2 - self.meanFace,
                                                            rcond = None)
        frames = []
        for f in range(1, 101):
            blend = f/100
            # Get eigenvalues by solving V @ U = Target
            reconstruct = self.V[:,:components] @ ( blend * values1 + ( 1 - blend) * values2)
            # Image as squared array
            reconstruct = np.reshape(reconstruct + self.meanFace,(128,128))
            # Return result as an image, as well as the residual error
            resIm = Image.fromarray(np.abs(reconstruct).astype(np.uint8)).convert("L")
            # Append image to the list of frames
            frames.append(resIm)

        # Save animation
        frames[0].save(name,
                       duration=t,
                       loop=0,
                       save_all=True, append_images=frames[1:], optimize=False)
        
        print(residuals1)

        return name