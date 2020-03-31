// To compile:
//TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
//TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
//g++ -std=c++11 -shared fast_mat_mul.cc -o fast_mat_mul.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
// if on a mac add this to the end of the final command: -undefined dynamic_lookup
// note: it must be tensorflow 2.0

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/common_runtime/dma_helper.h"

#include "mkl.h"

using namespace tensorflow;

REGISTER_OP("FastMatMul")
    .Input("a_matrix: float")
    .Input("b_matrix: float")
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

    const CBLAS_LAYOUT layout = CblasColMajor;
    const CBLAS_TRANSPOSE transa = CblasNoTrans;
    const CBLAS_TRANSPOSE transb = CblasNoTrans;
    const MKL_INT m = output_shape.dim_size(0);
    const MKL_INT n = output_shape.dim_size(1);
    const MKL_INT k = A_shape.dim_size(1);
    const MKL_INT lda = A_shape.dim_size(0);
    const MKL_INT ldb = B_shape.dim_size(0);
    const MKL_INT ldc = output_shape.dim_size(0);
    const float alpha = 1.0;
    const float beta = 0.0;
    const float* a = static_cast<const float *>(DMAHelper::base(&A_matrix));
    const float* b = static_cast<const float *>(DMAHelper::base(&B_matrix));
    float* c = static_cast<float *>(DMAHelper::base(output));
    cblas_sgemm(layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

//    const float* ptr = reinterpret_cast<const float*>(output->tensor_data().data());
//    std::cout<< ptr[0] <<std::endl;
//
//    float *ptr2 = static_cast<float *>(DMAHelper::base(output));
//    ptr2[0] = 7;
//    std::cout<< ptr2[0] <<std::endl;

    }
};
