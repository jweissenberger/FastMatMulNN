# FastMatMulNN

## [Here is a link to our paper](https://dl.acm.org/doi/10.1145/3458744.3474050) which was published in the ICPP International Conference on Parallel Processing Workshop in August 2021

This repository is a collection of the code, notebooks and results of our research into fast matrix multiplication algorithms in neural networks. Classic matrix multiply takes O(n^3) operations but algorithms like Strassen's and Bini's take less (about O(n^2.8) operations) while also introducing a small amount of error into the result. The goal of this research is to see if this small error in the result has an effect on the accuracy of neural networks during training. 
