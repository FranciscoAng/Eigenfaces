from CustomLib.FaceProcessor import ImageProcessor
from CustomLib.UI import UI

DATASET_DIR = "assets/rawdata/"
OUTPUT_DIR = "assets/output/"

if __name__ == "__main__":
    
    # Create Instances
    image_processor = ImageProcessor()
    #image_processor.get_dataset(DATASET_DIR)
    #image_processor.compute_mean()
    #image_processor.compute_eigV()
    window = UI(image_processor)

    # Start the GUI
    window.open()