# DocILE-Competition
The code of our system on the ICDAR-2023 DocILE competition.

# News
- **2023-06-09** All scripts and log files have been released!
- **2023-06-05** Finetuning code of GraphDoc model can be fetched here: [GraphDoc](https://github.com/ZZR8066/GraphDoc).
- **2023-06-05** Due to the complexity of the overall project, more experiment details would be made open-sourced in a few days. (Hopefully before 06/11/2023)

# Python Env Install
Note that some libs may be unnecessary.
```shell
pip install -r requirements.txt 
```

# How to Run
KILE and LIR tasks have the same directory structure. Here is an explanation of how to use the KILE task code.
* **Pre-train**: the code and data processing procedures used for pre-training are located in `./pretrain`, and we also provide the pre-trained models in `./model/pretrained_model`.
* **Processed OCR results**: during the fine-tuning process, the processed OCR results can be obtained by running the following command:
  ```shell
  cd ./ocr_dataprocess/scripts
  ./process_ocr.sh
  ```
* **Training and test data**: to generate training data and test data, you can use the following commands:
  ```shell
  # train data
  cd ./KILE/train_dataprocess
  ./process_training_set.sh

  # test data
  cd ./KILE/val_test_dataprocess
  ./process_test_set.sh
  ```
* **Train and test**: the train and test scripts can be found in the `./KILE/finetune`, and we also provide our training logs and testing results on different models.
* **Post-processing**: after obtaining the model classification and merge results, post-processing can be performed.
  ```shell
  cd ./KILE/post_process
  ./process_result_split_instance.sh
  ```
* **Model ensemble**: after selecting the models to be used for model ensemble, the following command can be used:
  ```shell
  cd ./KILE/model_ensemble
  ./ensemble_test.sh
  ```
  We also present the performance of four ensemble schemes on text box classification. 

# Pretrained Model

* We have released two pretrained GraphDoc model on DocILE unlabelled dataset, plus one swin transformer model. 
* Download them on `/model/pretrained_model` folder.
