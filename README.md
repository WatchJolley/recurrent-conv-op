# Recurrent Convolution - Tensorflow Op

A custom tensorflow op to achieve recurrent convolution as detailed in [Evolving Robust, Deliberate Motion Planning With a Shallow Convolutional Neural Network](https://www.mitpressjournals.org/doi/pdf/10.1162/isal_a_00099)

## About

Recurrent convolution can be achieved with standard TensorFlow operations. However, due to multiple cycles of convolution operations, the data will be passed back and forth across the CPU and GPU; this causes unnecessary latency. In this implementation, the input data is passed to the GPU once, then computed, and passed back.

The test provided uses a 20x20 input matrix with a 3x3 kernel. The current implementation only allows an input matrix; future work should focus on the compatibility of tensor inputs.

## Usage

The provided python test (`recurrent_tests.py`) allows experimentation with the op. The line:

```python
Wx_recurrent = recurrent_module.recurrent(d_input, d_W, 50)
```
Calls the op with the parameters:

* input matrix
* kernel
* num of cycles

## Notes

The main usage for this was to be incorporated into [Deep Neuroevolution](https://github.com/uber-research/deep-neuroevolution). Currently this is not functional.