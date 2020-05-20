#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <algorithm>


using namespace tensorflow;

/*
    Register Recurrent operation
*/

REGISTER_OP("Recurrent")
  .Input("input: double")
  .Input("weights: double")
  .Attr("iter: int = 1")
  .Output("output: double")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });

/*
    Recurrent Operation CPU
*/

class RecurrentOpCPU : public OpKernel {
public:
  explicit RecurrentOpCPU(OpKernelConstruction* context) : OpKernel(context) {
    // get the index of the value to preserve
    OP_REQUIRES_OK(context,
                   context->GetAttr("iter", &iter));

    // check that preserve_index is positive
    OP_REQUIRES(context, iter >= 0,
                errors::InvalidArgument("Need preserve_index >= 0, got ",
                                        iter));
  }
  
  void Compute(OpKernelContext* context) override {
    printf("Using Recurrent Module of CPU \n");

    // get the input tensor
    const Tensor& input = context->input(0);
    
    // get the weight tensor
    const Tensor& weights = context->input(1);
    
    // check shapes of input and weights
    const TensorShape& input_shape = input.shape();
    const TensorShape& weights_shape = weights.shape();
    
    // check that inputs are two dimensional-------------
    DCHECK_EQ(input_shape.dims(),   3);
    DCHECK_EQ(weights_shape.dims(), 3);
    //---------------------------------------------------
    
    const int batch_samples = input_shape.dim_size(input_shape.dims() - 1);
    printf("batch_samples %d\n", batch_samples);

    // do weight batches match input batches ------------
    DCHECK_EQ(batch_samples, weights_shape.dim_size(weights_shape.dims() - 1));
    //---------------------------------------------------

    const int input_feature_width = input_shape.dim_size(input_shape.dims() - 2);
    printf("input_feature_width %d\n", input_feature_width);

    const int input_feature_height = input_shape.dim_size(input_shape.dims() - 3);
    printf("input_feature_height %d\n", input_feature_height);

    // check input is square-----------------------------
    DCHECK_EQ(input_feature_width, input_feature_height);
    //---------------------------------------------------

    const int kernal_size = weights_shape.dim_size(weights_shape.dims() - 2);
    printf("kernal_size %d\n", kernal_size);

    const int kernal_depth = weights_shape.dim_size(weights_shape.dims() - 1);
    printf("kernal_depth %d\n", kernal_depth);

    // create output shape
    TensorShape output_shape;
    printf("batch_samples: %d\n", batch_samples);
    printf("kernal_size: %d\n", kernal_size);

    output_shape.AddDim(batch_samples);
    output_shape.AddDim(input_feature_width);
    output_shape.AddDim(input_feature_width);

    // create output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    
    // get the corresponding Eigen tensors for data access
    auto input_tensor   = input.flat<double>();
    auto weights_tensor = weights.flat<double>();
    auto output_tensor  = output->flat<double>();

    auto *inputs_  = &input_tensor(0);
    auto *weights_ = &weights_tensor(0);
    double *output_= &output_tensor(0);


    for (int ix_sample = 0; ix_sample < batch_samples; ix_sample++) {
        int batch     = ix_sample * (input_feature_width * input_feature_width);
        int strtPos = -(kernal_size/2);
        int endPos  =   kernal_size - abs(strtPos);

        // init output
        double outputClone_[input_feature_width*input_feature_width];
        for (int pos = 0; pos < input_feature_width*input_feature_width; pos++)
        {
            int p = pos + (ix_sample * (input_feature_width * input_feature_width));
            output_[p] = inputs_[p];
            outputClone_[pos] = inputs_[p];
        }

        for(int i = 0; i < iter; i++) {

            for(int x=0; x < input_feature_width; x++) for(int y=0; y < input_feature_width; y++)   {
              int row       = x * input_feature_width;
              int postion   = batch + row + y; 
              auto cell     = output_[postion];
              int pointer   = (ix_sample * (kernal_size*kernal_size));

            // filters around outputs---------------------------------------------------
            for(int fx=strtPos; fx < endPos; fx++) {
                for(int fy=strtPos; fy < endPos; fy++)   {

                    int newX = x + fx;
                    int newY = y + fy;

                    if (( newX < input_feature_width) && ( newX >= 0) && ( newY < input_feature_width) && ( newY >= 0)) {
                        int newrow   = newX * input_feature_width;
                        int clonePos = newrow + newY;
                        int newpos   = batch + clonePos; 

                        auto weight = weights_[pointer]/100; 

                        outputClone_[clonePos] += (cell * weight );

                    }
                    pointer += 1;
                }
            }
            //--------------------------------------------------------------------------

        }

        for (int pos = 0; pos < input_feature_width*input_feature_width; pos++)
        {
            output_[batch + pos] = std::max(outputClone_[pos], 0.0);
        }

        }
    }

  }
private:
    int iter;
};

REGISTER_KERNEL_BUILDER(Name("Recurrent").Device(DEVICE_CPU), RecurrentOpCPU);


// -------------------------------------------------------------------------------------------

/*
    Recurrent Operation GPU
*/

void RecurrentKernelLauncher(
        const double* inputs, 
        const double* weights,
        const int iteration,
        const int batch_samples, 
        const int units, 
        const int input_feature_width,
        const int kernal_width,
        double* output
    );


class RecurrentOpGPU : public OpKernel {
public:
  explicit RecurrentOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    // get the index of the value to preserve
    OP_REQUIRES_OK(context,
                   context->GetAttr("iter", &iter));

    // check that preserve_index is positive
    OP_REQUIRES(context, iter >= 0,
                errors::InvalidArgument("Need iteration number >= 0, received ",
                                        iter));
  }
  
  void Compute(OpKernelContext* context) override {
    printf("Using Recurrent Module of GPU \n");

    // get the input tensor
    const Tensor& input = context->input(0);
    
    // get the weight tensor
    const Tensor& weights = context->input(1);
    
    // check shapes of input and weights
    const TensorShape& input_shape = input.shape();
    const TensorShape& weights_shape = weights.shape();
    
    // check that inputs are two dimensional
    DCHECK_EQ(input_shape.dims(), 3);
    DCHECK_EQ(weights_shape.dims(), 3);
    
    const int batch_samples = input_shape.dim_size(input_shape.dims()-1);
    printf("batch_samples %d\n", batch_samples);

    const int input_feature_width = input_shape.dim_size(input_shape.dims()-2);
    printf("input_feature_width %d\n", input_feature_width);

    const int kernal_width = weights_shape.dim_size(weights_shape.dims()-2);
    printf("kernal_width %d\n", kernal_width);

    const int units = input_feature_width * input_feature_width;
    printf("units %d\n", units);

    // check input width matches weights height 
    DCHECK_EQ(input_feature_width, input_shape.dim_size(input_shape.dims()-2));

    const int iterations= iter;

    // create output shape
    TensorShape output_shape;
    //printf("batch_samples: %d\n", batch_samples);
    //printf("units: %d\n", units);

    output_shape.AddDim(batch_samples);
    output_shape.AddDim(input_feature_width);
    output_shape.AddDim(input_feature_width);
            
    // create output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    
    auto f_input    = input.flat<double>();
    auto f_weights  = weights.flat<double>();
    auto f_output   = output->template flat<double>();

    RecurrentKernelLauncher(
            f_input.data(), 
            f_weights.data(),
            iterations,
            batch_samples, 
            units, 
            input_feature_width,
            kernal_width,
            f_output.data()
        );
  }
private:
    int iter;
};

REGISTER_KERNEL_BUILDER(Name("Recurrent").Device(DEVICE_GPU), RecurrentOpGPU);


// -------------------------------------------------------------------------------------------