# python env install

1. pip install -r requirements.txt 
2. Note that some libs may be unnecessary, you can use your own environment if `PyMuPDF, numpy, cv2` exists.

# sentence bert model download

We use the sentence bert model shared in [huggingface](https://huggingface.co/sentence-transformers/roberta-base-nli-stsb-mean-tokens)

# scripts explanation

1. Excute `subprocess_ocr_line.sh` to render pdf into images and filter out low-quality pages.
2. Merge seperate output numpy files into a single .npy file. (np.load then list.extend then np.save)
3. Excute `subprocess_sentemb_line.sh` to extract sentence embeddings of text lines in each valid document pages.
4. Note that you can make serveral copies of .sh file or use the `--use_mp` option to process data parallelly.