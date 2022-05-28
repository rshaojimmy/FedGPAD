# TNNLS2022-FedGPAD
Pytorch codes for Federated Generalized Face Presentation Attack  Detection <a href=https://arxiv.org/pdf/2104.06595.pdf> (pdf) </a> in TNNLS 2022.


A face presentation attack detection (fPAD)  model with good generalization can be obtained when it is trained with face images from different input distributions and different types of spoof attacks. In reality, training data are not directly shared between data owners due to legal and privacy issues. In this paper, with the motivation of circumventing this challenge, we propose a Federated Generalized Face Presentation Attack Detection (FedGPAD) framework, studying the generalization issue of federating learning for fPAD in a data privacy preserving way.

<img src="FedGPAD.jpg" width="900">


In the proposed framework, each data owner (referred to as data centers) locally trains its own fPAD model. A server learns a global fPAD model by iteratively aggregating model updates from all data centers without accessing private data in each of them. To equip the aggregated fPAD model in the server with better generalization ability to unseen attacks from users, a federated domain disentanglement strategy is introduced in FedGPAD, which treats each data center as one domain and decomposes the fPAD model into domain-invariant and domain-specific parts in each data center. Two parts disentangle the domain-invariant and domain-specific features from images in each local data center, respectively. Several losses (i.e., cross-entropy classiﬁcation loss, depth estimation loss, image reconstruction loss, and feature difference loss) are proposed in this federated domain disentanglement strategy. A server learns a global fPAD model by only aggregating domain invariant parts of the fPAD models from data centers and thus a more generalized fPAD model can be aggregated in server in a data privacy preserving manner.



# Setup

* Prerequisites: Python3.6, pytorch=1.2, Numpy, libmr

* The source code folders:

  1. "models": Contains the network architectures of proposed network and process of federated learning. 
  3. "core": Contains the training and testing files. train_FedAvgDep.py contains the process of FedGPAD.
  4. "datasets": Contains datasets
  5. "misc": Contains initialization and some preprocessing functions.
  
# Training

To run the train file: python main.py --run_type Train

# Testing

To run the test file: python main.py --run_type Test --snapshotnum 10 (or converged epoch)

It will generate a .h5 file that contains the score for each frame. Then, we use these scores to calculate the AUC, EER, and HTER.

# Acknowledge
Please kindly cite this paper in your publications if it helps your research:
```
@article{shao2021federated,
  title={Federated Generalized Face Presentation Attack Detection},
  author={Shao, Rui and Perera, Pramuditha and Yuen, Pong C and Patel, Vishal M},
  journal={IEEE Transactions on Neural Networks and Learning Systems (TNNLS)},
  year={2022},
  organization={IEEE}
}
```

Contact: rshaojimmy@gmail.com
