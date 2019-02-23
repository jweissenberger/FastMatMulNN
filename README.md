# FastMatMulNN

This repository is a running collection of the code, notebooks and results of my research into fast matrix multiplication algorithms into neural networks. Classic matrix multiply takes O(n^3) but algorithms like Strassen's and Bini's take less (about O(n^2.8)) while also introducing a small amount of error into the result. The goal of this research is to see if this small error in the result has an effect on the accuracy of neural networks during training. Initial results look promising and can be seen in the results notebook and text files. 
