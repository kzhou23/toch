# TOCH: Spatio-Temporal Object-to-Hand Correspondence for Motion Refinement

Repo for **"TOCH: Spatio-Temporal Object-to-Hand Correspondence for Motion Refinement, ECCV'22 (Poster)"** \
[[Paper]](http://virtualhumans.mpi-inf.mpg.de/papers/zhou22toch/toch.pdf) [[Project Page]](http://virtualhumans.mpi-inf.mpg.de/toch)

## Environment
We recommend running the code in conda environment:
```shell
conda create -n toch python=3.7
conda activate toch 
```
Clone the repository and install main dependencies with
```shell
git clone https://github.com/kzhou23/toch.git && cd toch 
pip install -r requirements.txt
```
We additionally require the following libraries：
- [MPI-IS Mesh Processing Library](https://github.com/MPI-IS/mesh)
- [Manopth layer for PyTorch](https://github.com/hassony2/manopth)

Please check the respective instructions for downloading and installation.

## Data Preparation
### GRAB Dataset
1. Download the raw GRAB dataset and SMPL-X models by following instructions [here](https://github.com/otaheri/GRAB).
2. Clone the GRAB repository and copy its subfolder:
```shell
git clone https://github.com/otaheri/GRAB.git
cp -r GRAB/tools toch/data/grab/
```
3. Run our pre-processing code:
```shell
python data/grab/preprocessing.py --grab_path $RAW_GRAB_FOLDER \
                                  --smplx_path $SMPLX_MODEL_FOLDER \
                                  --mano_path $MANO_MODEL_FOLDER \
                                  --out_path $PROCESSED_GRAB_FOLDER
python data/grab/compute_hand_obj_corr.py --grab_path $RAW_GRAB_FOLDER \
                                          --data_path $PROCESSED_GRAB_FOLDER \
                                          --mano_path $MANO_MODEL_FOLDER \
                                          --num_proc 50
```

### Custom Dataset
TODO

## Training
Train the model with
```shell
python train.py --num_gpu 3 --data_path $PROCESSED_GRAB_FOLDER
```
The model checkpoint will be saved under `./ckpt` by default. Feel free to explore the available training options.
## Inference
### GRAB Dataset
You can refine a sequence from the pre-processed GRAB dataset with
```shell
python scripts/reconstruct_grab_seq.py --grab_path $RAW_GRAB_FOLDER \
                                       --ckpt_path $PRETRAINED_MODEL_PATH \
                                       --mano_path $MANO_MODEL_FOLDER \
                                       --seq_path $INPUT_SEQUENCE_PATH
```
The output meshes will be saved under `./recon_results` by default.

### Custom Dataset
TODO

## Citation
```bibtex
@inproceedings{zhou2022toch,
    title = {TOCH: Spatio-Temporal Object Correspondence to Hand for Motion Refinement},
    author = {Zhou, Keyang and Bhatnagar, Bharat Lal and Lenssen, Jan Eric and Pons-Moll, Gerard},
    booktitle = {European Conference on Computer Vision ({ECCV})},
    month = {October},
    organization = {{Springer}},
    year = {2022},
}
