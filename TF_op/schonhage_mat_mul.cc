// To compile:
//TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
//TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
// if on a mac add this to the end of the final command: -undefined dynamic_lookup
// note: it must be tensorflow 2.0

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/common_runtime/dma_helper.h"

// include arbenson-fast-matmul stuff
#include "linalg.hpp"
#include "schonhage333_21_117_approx.hpp"

using namespace tensorflow;

REGISTER_OP("FastMatMul")
    .Input("a_matrix: float")
    .Input("b_matrix: float")
    .Input("epsilon: double")
    .Input("steps: int32")
    .Output("fast_mat_mul: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle A_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &A_shape)); // get shape of input matrix A

    shape_inference::ShapeHandle B_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &B_shape)); // get shape of input matrix B

    // get dimensions for the input matrices
    shape_inference::DimensionHandle A_rows = c->Dim(A_shape, 0);
    shape_inference::DimensionHandle B_cols = c->Dim(B_shape, 1);

    // set output matrix
    c->set_output(0, c->Matrix(A_rows, B_cols));
    return Status::OK();
  });

class FastMatMulOp : public OpKernel {
public:
  /// \brief Constructor.
  /// \param context
  explicit FastMatMulOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {


    // get the inputs
    const Tensor& A_matrix = context->input(0);
    const Tensor& B_matrix = context->input(1);

    // check shapes of inputs
    const TensorShape& A_shape = A_matrix.shape();
    const TensorShape& B_shape = B_matrix.shape();

    // check that they're matrices
    DCHECK_EQ(A_shape.dims(), 2);
    DCHECK_EQ(B_shape.dims(), 2);

    // check they can multiply
    DCHECK_EQ(A_shape.dim_size(1), B_shape.dim_size(0));

    // create output shape
    TensorShape output_shape;
    output_shape.AddDim(A_shape.dim_size(0));
    output_shape.AddDim(B_shape.dim_size(1));

    // create output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    // cast tensor data as Matrix (class defined by arbenson)
    const float* a = static_cast<const float *>(DMAHelper::base(&A_matrix));
    const float* b = static_cast<const float *>(DMAHelper::base(&B_matrix));
    float* c = static_cast<float *>(DMAHelper::base(output));
    Matrix<float> A = Matrix<float>(a, A_shape.dim_size(0), A_shape.dim_size(0), A_shape.dim_size(1));
    Matrix<float> B = Matrix<float>(b, B_shape.dim_size(0), B_shape.dim_size(0), B_shape.dim_size(1));
    Matrix<float> C = Matrix<float>(c, output->dim_size(0), output->dim_size(0), output->dim_size(1));
    auto numsteps_tmp = context->input(4).scalar<int>();
    int numsteps = numsteps_tmp(0); // number of recursive steps
    auto epsilon_tmp = context->input(3).scalar<float>();
    double epsilon = epsilon_tmp(0); // error parameter (to be tuned for numsteps)
    
    // call Schonhage's matmul
    schonhage333_21_117_approx::FastMatmul(A, B, C, numsteps, epsilon);

//    const float* ptr = reinterpret_cast<const float*>(output->tensor_data().data());
//    std::cout<< ptr[0] <<std::endl;
//
//    float *ptr2 = static_cast<float *>(DMAHelper::base(output));
//    ptr2[0] = 7;
//    std::cout<< ptr2[0] <<std::endl;

    }
};

REGISTER_KERNEL_BUILDER(Name("FastMatMul").Device(DEVICE_CPU), FastMatMulOp);
