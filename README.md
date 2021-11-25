# lung-tumor-segmentation

A project I'm currently working on just for fun and educational purposes.
In this project, I used the lung tumor data from the [Medical Decathlon competition]( https://drive.google.com/file/d/1I1LR7XjyEZ-VBQ-Xruh31V7xExMjlVvi/view?usp=sharing)

## Results
    
    
### Using this project

In order for this project to log things properly, you need to:
   - run **wandb login** in your terminal and provide your own key

To run this project yourself:
1. Clone the project
2. Install the packages from the requirements.txt file in your python env.
3. Download the medical decathlon lung tumor data and extract it to a folder.
4. Run: python preprocessing.py --input_data_dir <extraction_path>/imagesTr --input_labels_dir <extraction_path>/labelsTr.
(If you want to try this out on your own data - the expected format is Nifty files for both the scan and the mask data. The scan and corresponding mask must have the same name and be in different folders)