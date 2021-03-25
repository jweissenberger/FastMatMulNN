
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/common_runtime/dma_helper.h"

// include arbenson-fast-matmul stuff
#include "linalg.hpp"
#include "smirnov722_22_198_approx.hpp"

using namespace tensorflow;

REGISTER_OP("FastMatMul")
    .Attr("epsilon: float")
    .Attr("steps: int")
    .Attr("numthreads: int")
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
    /// brief Constructor.
    /// param context
    explicit FastMatMulOp(OpKernelConstruction* context) : OpKernel(context) {
        // get attrs

        OP_REQUIRES_OK(context, context->GetAttr("steps", &numsteps_));
        OP_REQUIRES_OK(context,context->GetAttr("epsilon", &epsilon_));
        OP_REQUIRES_OK(context,context->GetAttr("numthreads", &numthreads_));
    }

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
        Matrix<float> A = Matrix<float>(b, B_shape.dim_size(1), B_shape.dim_size(1), B_shape.dim_size(0));
        Matrix<float> B = Matrix<float>(a, A_shape.dim_size(1), A_shape.dim_size(1), A_shape.dim_size(0));
        Matrix<float> C = Matrix<float>(c, output->dim_size(1), output->dim_size(1), output->dim_size(0));

        // call smirnov722's matmul
        smirnov722_22_198_approx::FastMatmul(A, B, C, numsteps_, epsilon_, numthreads_);

//    const float* ptr = reinterpret_cast<const float*>(output->tensor_data().data());
//    std::cout<< ptr[0] <<std::endl;

//    float *ptr2 = static_cast<float *>(DMAHelper::base(output));
//    ptr2[0] = 7;
//    std::cout<< ptr2[0] <<std::endl;

    }

    private:
        int numsteps_;
        float epsilon_;
        int numthreads_;

    };

REGISTER_KERNEL_BUILDER(Name("FastMatMul").Device(DEVICE_CPU), FastMatMulOp);
