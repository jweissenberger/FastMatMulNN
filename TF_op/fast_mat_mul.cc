#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/common_runtime/dma_helper.h"

using namespace tensorflow;

REGISTER_OP("ClassicMatMul")
    .Input("a_matrix: float")
    .Input("b_matrix: float")
    .Output("classic_mat_mul: float")
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

class ClassicMatMulOp : public OpKernel {
public:
  /// \brief Constructor.
  /// \param context
  explicit ClassicMatMulOp(OpKernelConstruction* context) : OpKernel(context) {}

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

    // get the corresponding Eigen tensors for data access
    auto A_tensor = A_matrix.matrix<float>();
    auto B_tensor = B_matrix.matrix<float>();
    auto output_tensor = output->matrix<float>();

    const CBLAS_LAYOUT layout = CblasColMajor;
    const CBLAS_TRANSPOSE transa = 'N';
    const CBLAS_TRANSPOSE transb = 'N';
    const MKL_INT m = output_shape.dim_size(0);
    const MKL_INT n = output_shape.dim_size(1);
    const MKL_INT k = A_shape.dim_size(1);
    const MKL_INT lda = A_shape.dim_size(0);
    const MKL_INT ldb = B_shape.dim_size(0);
    const MKL_INT ldc = output_shape.dim_size(0);
    const float alpha = 1.0;
    const float beta = 0.0;
    const float* a = reinterpret_cast<const float*>(A_tensor->tensor_data().data());
    const float* b = reinterpret_cast<const float*>(B_tensor->tensor_data().data());
    const float* c = static_cast<float *>(DMAHelper::base(output));
    maybe let's understand better the three lines above, DMAHelper vs data() and reinterpret_cast vs static_cast

//    std::cout<<A_shape.dim_size(0) << A_shape.dim_size(1) <<std::endl;
//    std::cout<<B_shape.dim_size(0) << B_shape.dim_size(1)<<std::endl;
//    std::cout<<output_shape.dim_size(0) << output_shape.dim_size(1)<<std::endl;


//    const float* ptr = reinterpret_cast<const float*>(output->tensor_data().data());
//    std::cout<< ptr[0] <<std::endl;
//
//    float *ptr2 = static_cast<float *>(DMAHelper::base(output));
//    ptr2[0] = 7;
//    std::cout<< ptr2[0] <<std::endl;

    }
}