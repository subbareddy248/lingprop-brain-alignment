# Joint processing of linguistic properties in brains and language models

[Joint processing of linguistic properties in brains and language models](https://arxiv.org/abs/2212.08094), Subba Reddy Oota, Manish Gupta and Mariya Toneva, NeurIPS-2023

![screenshot](architecture.png)

[21<sup>st</sup> year_narratives_listening_dataset](https://figshare.com/articles/dataset/BOLD5000_Release_2_0/14456124)

**21<sup>st</sup> year dataset statistics:**
  
- 18 subjects
- fMRI brain recordings
- 8267 words
- 2226 TRs (Time Repetition)
- TR = 1.5 secs

**How to download 21<sup>st</sup> year dataset**
  
* Datalad can be installed using pip

```bash
python -m pip install datalad
```
- It is highly recommended to configure Git before using DataLad. Set both 'user.name' and 'user.email' configuration variables.
```bash
- git config --global user.name "username"
- git config --global user.email emailid
```
- git-annex installation is required for downloading the dataset
```bash
sudo apt-get install git-annex
```

**Download the dataset using datalad**
```bash
datalad install https://datasets.datalad.org/labs/hasson/narratives/derivatives/afni-nosmooth
```

**Download each subject data (considered the fsaverage6) using bash script**
```bash
cd afni-nosmooth
bash download_data.sh
```

**Extract stimuli representations using bert model with context length 20**
- Narratives 21st-year Dataset
```bash
python extract_features_words.py --input_file ./Narratives/21styear_align.csv --model bart-base --sequence_length 20 --output_file bert-base-20
```

**To build voxelwise encoding model for different stimuli representations**
- five arguments are passed as input: subject_number, #layers, stimulus vector, context length, and output directory
```
cd brain_predictions
python brain_predictions_21styear_text.py 1 12 bert_conext20_21styear.npy 20 output_predictions
```

## Poster
[Poster](https://drive.google.com/file/d/1FOpiNJpXma3mlOK0F9nLhcJpJaJSQWsS/view?usp=sharing)

## Slides
[slides](https://drive.google.com/file/d/1dczwbzrHmfitXSINBFRo5B3QcZ_5eGck/view?usp=sharing)

## Video
[Video](https://nips.cc/virtual/2023/poster/72702)

## For Citation of our work
```
@inproceedings{oota2022joint,
  title={Joint processing of linguistic properties in brains and language models},
  author={Oota, Subba Reddy and Gupta, Manish and Toneva, Mariya},
  booktitle={Proceedings of the Thirty-seventh Conference on Neural Information Processing Systems },
  year={2023}
}
```
