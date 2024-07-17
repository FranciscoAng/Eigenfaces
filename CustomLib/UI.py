from CustomLib.FaceProcessor import ImageProcessor

import tkinter as tk
from tkinter import ttk, filedialog, Label
from PIL import Image, ImageTk
import numpy as np

class UI:
    # Initialisation
    def __init__(self, image_processor : ImageProcessor):

        self.image_processor = image_processor
        self.root = tk.Tk()

        # Interface settings
        self.root.title("Eigenfaces and FaceMorph")
        self.frm = ttk.Frame(self.root, padding=10)
        self.frm.grid()

        # Deffault images
        self.image_blank = ImageTk.PhotoImage(file = "assets/blank_image.jpg")
        self.image_result = ImageTk.PhotoImage(file = "assets/blank_result.jpg")

        # Keep references to avoid garbage collection
        self.images = {
            "blank": self.image_blank,
            "result": self.image_result,
            "image1": None,
            "image2": None
        }
        self.squared_error = None

        # 1st row
        ttk.Label(self.frm, text="Face Morpher V1").grid(column=0, row=0)

        # 3rd row
        self.image_label1 = ttk.Label(self.frm, image=self.image_blank)
        self.image_label1.grid(column= 1, row=2)
        self.image_label2 = ttk.Label(self.frm, image=self.image_blank)
        self.image_label2.grid(column= 2, row=2)
        ttk.Label(self.frm, text="-->").grid(column= 3, row=2)
        self.image_label3 = ttk.Label(self.frm, image=self.image_result)
        self.image_label3.grid(column= 4, row=2)

        # 2nd row
        ttk.Button(self.frm, text="Upload Image 1",
                   command = lambda: self.upload_image("image1", self.image_label1)
                   ).grid(column=1, row=1)
        ttk.Button(self.frm, text="Upload Image 2",
                   command = lambda: self.upload_image("image2", self.image_label2)
                   ).grid(column=2, row=1)

        # 4th row
        ttk.Button(self.frm, text="Delete Image 1",
                   command = lambda : self.delete_image("image1", self.image_label1)
                   ).grid(column=1, row=3)
        ttk.Button(self.frm, text="Delete Image 2",
                   command = lambda : self.delete_image("image2", self.image_label2)
                   ).grid(column=2, row=3)

        # 5th row
        ttk.Button(self.frm, text="Show 'SVD'", command=self.show_svd_parameters).grid(column=0, row=4)
        ttk.Button(self.frm, text="Show Graph", command=self.show_graph).grid(column=1, row=4)
        ttk.Button(self.frm, text="Show Reconstructed Images", command=self.show_reconstructed_images).grid(column=2, row=4)
        ttk.Button(self.frm, text="Show Reconstructed Gifs", command = self.show_reconstructed_gifs).grid(column=3, row=4)

        # 6th row
        ttk.Button(self.frm, text="Quit", command=self.root.destroy).grid(column=4, row=5)

    # Opens the GUI
    def open(self):
        self.root.mainloop()

    # Function to upload an Image
    def upload_image1(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path and self.images["image1"] is None:
            new_image = self.image_processor.import_image(file_path, is_dataset=False)
            if new_image is None:
                return
            
            self.images["image1"] = new_image
            new_image = self.array_to_photoimage(new_image)
            self.image_label1.configure(image = self.images["image1"])
            #self.images["image1"] = new_image # Keep references to avoid garbage collection
            #self.image1_uploded = True

    
    def upload_image2(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path and self.images["image2"] is None:
            new_image = self.image_processor.import_image(file_path)
            new_image = self.array_to_photoimage(new_image)
            self.image_label2.configure(image=new_image)
            self.images["image2"] = new_image # Keep references to avoid garbage collection
            #self.image2_uploded = True

    # Function to upload an Image
    def upload_image(self, image_dict : str, lbl_display : tk.Label):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        #if file_path and img_data is None:
        if not file_path:
            return
        
        new_image = self.image_processor.import_image(file_path, is_dataset=False)
        if new_image is None:
            return
        
        self.images[image_dict] =  new_image.flatten() # Keep references to avoid garbage collection

        disp_img = self.array_to_photoimage(new_image)
        lbl_display.configure(image = disp_img)
        lbl_display.image = disp_img
        print(self.images[image_dict])

    # Function to delete an Image
    def delete_image1(self, number=1):
        #self.image_blank = ImageTk.PhotoImage(Image.open("assets/blank_image.jpg"))
        self.image_label1.configure(image = self.image_blank)
        self.images["image1"] = None
        #self.image_processor.remove_image(number)
        #self.image1_uploded = False
    def delete_image2(self, number=2):
        #self.image_blank = ImageTk.PhotoImage(Image.open("assets/blank_image.jpg"))
        self.image_label2.configure(image = self.image_blank)
        #self.image_processor.remove_image(number)
        #self.image2_uploded = False

    def delete_image(self, dict_idx : str, lbl_display: ttk.Label):
        lbl_display.configure(image = self.image_blank)
        self.images[dict_idx] = None

    # Converts an np.array into a photoimage
    def array_to_photoimage(self, array : np.ndarray):
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
        if self.images["image1"] is not None and self.images["image2"] is not None:
            self.gif_path = self.image_processor.blend_images(self.images["image1"], self.images["image2"], 1000)
        else:
            return
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