# Lung Tumor Segmentation

A project I'm currently working on just for fun and educational purposes.
In this project, I used the lung tumor data from the [Medical Decathlon competition](https://drive.google.com/file/d/1I1LR7XjyEZ-VBQ-Xruh31V7xExMjlVvi/view?usp=sharing)
    
### Using this project

In order for this project to log things properly, you need to:
   - run **wandb login** in your terminal and provide your own key

To run this project yourself:
1. Clone the project, cd into project directory and run pip install -e .
2. Install the packages from the requirements.txt file in your python env.
3. Download the medical decathlon lung tumor data and extract it to a folder.
4. Run: python preprocessing.py --input_data_dir <extraction_path>/imagesTr --input_labels_dir <extraction_path>/labelsTr --output_dir <path_to_output_dir>
(If you want to try this out on your own data - the expected format is Nifty for both the scan and the mask data. The scan and corresponding mask must have the same name and be in different folders)
5. Run: python train.py --preprocessed_input_dir <path_to_output_dir> (This is the output directory you provided in the previous step)
6. **Inference** : run python inference.py --path_to_ckpt <path_to_ckpt> path_to_ct_scan <path_to_nifty_ct_scan> --path_to_result_dir <output_dir>. 

My SegNet checkpoint can be downloaded from [this link](https://drive.google.com/file/d/1qlj4yZuEM2FoNzaXPFBG6Pjtl-A1mS1t/view?usp=sharing) and used with the inference script.

# Results
So far, using the architecture proposed in the [SegNet paper](https://arxiv.org/pdf/1511.00561.pdf), I reached nice results (0.88 dice score, 0.75 IoU on the validation set)

Example result:

![Side by Side](/images/sidebyside.png)

### Here is the prediction for an entire ct scan

https://user-images.githubusercontent.com/17112442/143877689-bd9221f4-ff81-4072-a359-e077b1d37b06.mp4

As we can see, it's not accurate but still a nice result :smile:

#### It is still a work in progress


