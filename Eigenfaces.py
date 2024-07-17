import sys
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, Label
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib import cm

IMAGE_COUNT = 150
NUMBER_OF_COMPONENTS = 5
AMOUNT_OF_FRAMES = 10
GIF_STRING = "test.gif"

class ImageProcessor:
    # Initializes the Image Processor
    def __init__(self, image_count=IMAGE_COUNT):
        self.imagefile1 = None
        self.imagefile2 = None
        self.x_prev_values = None
        self.components = 500   # Set the number of principal components to be used in further analysis
        self.image_count = image_count  # Initialize with the number of images to import
        self.original_images = self.import_images() # Import images and store them in the original_images list
        self.update_image_data() # Updates all images and data
        
    # Imports the already Database of Images
    def import_images(self):
        images = []
        for i in np.random.randint(1223, 5222, self.image_count): # Import a random selection of images based on the specified count
            try:
                img = np.fromfile(f"assets/rawdata/{i}", dtype=np.uint8).reshape((128, 128))   # Read the image file and reshape it to 128x128 pixels
                images.append(img)
            except:
                pass    # Skip files that cannot be read or reshaped
        print(f"Imported {len(images)} images")
        return images

    # Finds the smallest Image and returns that Size
    def find_min_size(self, images):
        min_rows = min_cols = sys.maxsize
        for image in images:    # Determine the minimum number of rows and columns across all images
            r, c = image.shape
            min_rows = min(min_rows, r)
            min_cols = min(min_cols, c)
        print(f"\n==> Least common image size: {min_rows} x {min_cols} pixels")
        return min_rows, min_cols

    # Recenter and crop each image to the smallest Size
    def recenter_images(self, images, min_rows, min_cols):
        return [self.recenter(image, min_rows, min_cols) for image in images] 

    # Performs operations on the images
    def plot_reconstructed_images(self, plotGraph=None, plotImages=None):
        # Project the data matrix onto the first two principal components
        num_components = NUMBER_OF_COMPONENTS
        Y = np.matmul(self.X, self.VT[:num_components, :].T)
        # Plot the singular values to visualize their decay
        if plotGraph:
            plt.plot(Y[:250, :], 'ro')
            plt.show()

        if (plotImages):
            fig, axs = plt.subplots(2, 2)
            Y_reduced = np.diag(np.abs(self.eigenvalues[:self.components])) @ self.U.T
            for i, ax in enumerate(axs.flat):
                ax.imshow(np.abs(Y_reduced[i, :]).reshape(128, 128), cmap=cm.gray)
            plt.show()

    # Create a GIF for that shows the morph effect
    def reconstruct_and_save_gifs(self):
        if self.imagefile1 and self.imagefile2:
            image1 = Image.open(self.imagefile1).convert("L").resize((128, 128))
            image2 = Image.open(self.imagefile2).convert("L").resize((128, 128))
            test_images = [np.array(image1).flatten(), np.array(image2).flatten()]
            means = self.X.mean(axis=0)

            for idx, test_image in enumerate(test_images):
                frames = []

                # Ensure self.U has the same number of rows as the number of elements in x
                if self.U.shape[0] != len(test_image):
                    raise ValueError(f"Incompatible dimensions: U has {self.U.shape[0]} rows, but x has {len(test_image)} elements")

                values, *_ = np.linalg.lstsq(self.U, test_image, rcond=None) #Least square calculation of the linear matrix / *_ ignores the rest of the output from np.linalg.lstsq
                for f in range(0, self.components, 1):
                    reconstruct = self.U[:, :f] @ values[:f] + means
                    res_image = np.abs(reconstruct).reshape(128, 128).astype(np.uint8)
                    frames.append(Image.fromarray(res_image).convert("P"))
                file_path = GIF_STRING
                frames[0].save(file_path, save_all=True, append_images=frames[1:], optimize=False, duration=100, loop=0)
            return file_path

    # Adds an image to the end of the raw data list and updates the other functions with respect of the new image
    def add_image(self, file_path, number):
        try:
            if number == 1:
                self.imagefile1 = file_path
            elif number == 2:
                self.imagefile2 = file_path
            else:
                print("Something went wrong, storing the image when adding")
            img_array = np.array(Image.open(file_path).convert("L").resize((128, 128)))
            self.original_images.append(img_array)
            self.update_image_data()  # Update all related data after adding a new image
            print(f"Added image: {file_path}")
            return img_array
        except Exception as e:
            print(f"Error adding image: {e}")

    # Removes an image
    def remove_image(self, number):
        try:
            image_nr = self.image_count - 1 + number #number of images + my added image
            if (number == 1):
                self.imagefile1 = None
            elif(number == 2):
                self.imagefile2 = None
            else:
                print("Error removing image")
            self.original_images.pop(image_nr) # Code for removeing the image
            self.update_image_data()  # Update all related data after adding a new image
            print(f"Removed image")
            return True
        except Exception as e:
            print(f"Error removing image: {e}. Maybe the Image is already removed")

    # Update when images are added
    def update_image_data(self):
        self.min_rows, self.min_cols = self.find_min_size(self.original_images) # Find the smallest dimensions across all images for consistent resizing
        self.recentered_images = self.recenter_images(self.original_images, self.min_rows, self.min_cols)   # Recenter and crop all images to the smallest dimensions found
        self.X = self.create_data_matrix(self.recentered_images)  # Create a data matrix from the recentered images, where each image is flattened into a 1D vector
        self.U, self.S, self.VT = self.compute_svd(self.X)   # Compute the Singular Value Decomposition (SVD) of the data matrix
        means = self.X.mean(axis=0)

        A = (self.X - means).T
        M = A.T @ A
        self.eigenvalues, self.eigenvectors = np.linalg.eig(M)
        sorted_indices = np.argsort(self.eigenvalues)[::-1]
        self.U = np.array([A @ self.eigenvectors[:, i] / np.linalg.norm(A @ self.eigenvectors[:, i]) for i in sorted_indices[:self.components]]).T

    # Perform Singular Value Decomposition (SVD) on the data matrix
    def compute_svd(self, X, plot=None):
        if X is not None:
            self.x_prev_values = X
        else:
            X = self.x_prev_values

        U, S, VT = np.linalg.svd(X, full_matrices=False)
        print(f"X: {X.shape}\nU: {U.shape}\nSigma: {S.shape}\nV^T: {VT.shape}")
        
        if (plot):
            plt.plot(S, 'ro')
            plt.show()
        return U, S, VT

    # Calculate the amount to crop from each side to center the image
    def recenter(self, image, min_rows, min_cols):
        try:
            if image.shape:
                r, c = image.shape
        except: 
            if image.size:
                r, c = image.size
        top = (r - min_rows) // 2
        left = (c - min_cols) // 2
        return image[top:top + min_rows, left:left + min_cols]

    # Flatten each image into a 1D vector and stack them into a data matrix
    def create_data_matrix(self, images): 
        matrix = np.array([image.flatten() for image in images])
        self.m =len(images)
        self.d = self.min_rows * self.min_cols
        self.X = np.reshape(images, (self.m, self.d))
        return matrix

class UI:
    # Initialisation
    def __init__(self, image_processor):
        self.image1_uploded = False
        self.image2_uploded = False
        self.image_processor = image_processor
        self.root = tk.Tk()
        self.root.title("Eigenfaces and FaceMorph")
        self.frm = ttk.Frame(self.root, padding=10)
        self.frm.grid()

        # 1st row
        ttk.Label(self.frm, text="Face Morpher V1").grid(column=0, row=0)

        # 2nd row
        ttk.Button(self.frm, text="Upload Image 1", command=self.upload_image1).grid(column=1, row=1)
        ttk.Button(self.frm, text="Upload Image 2", command=self.upload_image2).grid(column=2, row=1)

        # 3rd row
        self.image_blank = ImageTk.PhotoImage(Image.open("assets/blank_image.jpg"))
        self.image_result = ImageTk.PhotoImage(Image.open("assets/blank_result.jpg"))
        
        self.image_label1 = ttk.Label(self.frm, image=self.image_blank)
        self.image_label1.grid(column= 1, row=2)
        self.image_label2 = ttk.Label(self.frm, image=self.image_blank)
        self.image_label2.grid(column= 2, row=2)
        ttk.Label(self.frm, text="-->").grid(column= 3, row=2)
        self.image_label3 = ttk.Label(self.frm, image=self.image_result)
        self.image_label3.grid(column= 4, row=2)

        # Keep references to avoid garbage collection
        self.images = {
            "blank": self.image_blank,
            "result": self.image_result,
            "image1": None,
            "image2": None
        }

        # 4th row
        ttk.Button(self.frm, text="Delete Image 1", command=self.delete_image1).grid(column=1, row=3)
        ttk.Button(self.frm, text="Delete Image 2", command=self.delete_image2).grid(column=2, row=3)

        # 5th row
        ttk.Button(self.frm, text="Show 'SVD'", command=self.show_svd_parameters).grid(column=0, row=4)
        ttk.Button(self.frm, text="Show Graph", command=self.show_graph).grid(column=1, row=4)
        ttk.Button(self.frm, text="Show Reconstructed Images", command=self.show_reconstructed_images).grid(column=2, row=4)
        ttk.Button(self.frm, text="Show Reconstructed Gifs", command=self.show_reconstructed_gifs).grid(column=3, row=4)

        # 6th row
        ttk.Button(self.frm, text="Quit", command=self.root.destroy).grid(column=4, row=5)

    # Opens the GUI
    def open(self):
        self.root.mainloop()

    # Function to upload an Image
    def upload_image1(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path and not self.image1_uploded:
            new_image = self.image_processor.add_image(file_path, 1)
            new_image = self.array_to_photoimage(new_image)
            self.image_label1.configure(image=new_image)
            self.images["image1"] = new_image # Keep references to avoid garbage collection
            self.image1_uploded = True
    def upload_image2(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path and not self.image2_uploded:
            new_image = self.image_processor.add_image(file_path, 2)
            new_image = self.array_to_photoimage(new_image)
            self.image_label2.configure(image=new_image)
            self.images["image2"] = new_image # Keep references to avoid garbage collection
            self.image2_uploded = True

    # Function to delete an Image
    def delete_image1(self, number=1):
        self.image_blank = ImageTk.PhotoImage(Image.open("assets/blank_image.jpg"))
        self.image_label1.configure(image=self.image_blank)
        self.image_processor.remove_image(number)
        self.image1_uploded = False
    def delete_image2(self, number=2):
        self.image_blank = ImageTk.PhotoImage(Image.open("assets/blank_image.jpg"))
        self.image_label2.configure(image=self.image_blank)
        self.image_processor.remove_image(number)
        self.image2_uploded = False

    # Converts an np.array into a photoimage
    def array_to_photoimage(self, array):
        # Ensure array is in uint8 format and has 3 channels (RGB)
        if array.dtype != np.uint8:
            array = (array * 255).astype(np.uint8)
        if array.ndim == 2:
            array = np.stack((array,) * 3, axis=-1)  # Convert grayscale to RGB
        elif array.shape[2] == 4:
            array = array[..., :3]  # Remove alpha channel if present
        
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(array)
        
        # Convert PIL Image to PhotoImage
        return ImageTk.PhotoImage(pil_image)

    # Calls the SVD curve
    def show_svd_parameters(self):
        self.image_processor.compute_svd(X=None, plot=True)

    # Calls the Graph 
    def show_graph(self):
        self.image_processor.plot_reconstructed_images(plotGraph = True)

    # Calls the reconstructed_images function of the image_processor
    def show_reconstructed_images(self):
        self.image_processor.plot_reconstructed_images(plotImages = True)

    # Calls the show_reconstructed_gifs function of the image_processor
    def show_reconstructed_gifs(self):
        self.gif_path = self.image_processor.reconstruct_and_save_gifs()
        self.gif = Image.open(self.gif_path)
        self.frames = []

        try:
            while True:
                frame = ImageTk.PhotoImage(self.gif.copy())
                self.frames.append(frame)
                self.gif.seek(len(self.frames))  # Move to the next frame

        except EOFError:
            pass

        self.frame_count = len(self.frames)
        self.update_frame(0)  # Start the animation from the first frame

    # Updates the GIF frame in the UI window when called
    def update_frame(self, ind):
        frame = self.frames[ind]
        ind = (ind + 1) % self.frame_count  # Loop back to the first frame after the last frame
        self.image_label3.configure(image=frame)
        self.root.after(100, self.update_frame, ind)  # Use self.root instead of window

if __name__ == "__main__":
    
    # Create Instances
    image_processor = ImageProcessor()
    window = UI(image_processor)

    # Start the GUI
    window.open()
