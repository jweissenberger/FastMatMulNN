#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("ClassicMatMul")
    .Input("A_matrix: float")
    .Input("B_matrix: float")
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

    // classic mm
    for (int i = 0; i < A_shape.dim_size(0); i++){
        for (int j = 0; j < B_shape.dim_size(1); j++){
            for (int k = 0; j < B_shape.dim_size(1); j++){
                output_tensor(i, j) += A_tensor(i, k) * B_tensor(k, j);
            }
        }
    }

  }
};

REGISTER_KERNEL_BUILDER(Name("ClassicMatMul").Device(DEVICE_CPU), ClassicMatMulOp);