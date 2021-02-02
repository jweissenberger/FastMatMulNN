

def write_line(header, num_indent, code):
    header.write(' ' * 4 * num_indent + code + '\n')

def write_break(header, num_breaks = 1):
    header.write('\n' * num_breaks)

def writer(in_file, out_file):
    try:
        in_file = in_file.split('/')[-1]
        namespace_name = in_file.split('.')[0]
        algo_name = namespace_name.split('_')[0]
        print("Genrating code for %s" % out_file)
    except:
        raise Exception("USAGE: python opgen.py in_file out_file")

    with open(out_file,"w") as header:
        write_break(header)

        write_line(header,0,'#include "tensorflow/core/framework/op_kernel.h"')
        write_line(header,0,'#include "tensorflow/core/framework/tensor_shape.h"')
        write_line(header,0,'#include "tensorflow/core/platform/default/logging.h"')
        write_line(header,0,'#include "tensorflow/core/framework/shape_inference.h"')
        write_line(header,0,'#include "tensorflow/core/common_runtime/dma_helper.h"')
        write_break(header)

        write_line(header,0,'// include arbenson-fast-matmul stuff')
        write_line(header,0,'#include "linalg.hpp"')
        write_line(header,0,'#include "%s"' % in_file)
        write_break(header)

        write_line(header,0,'using namespace tensorflow;')
        write_break(header)

        write_line(header,0,'REGISTER_OP("FastMatMul")')
        write_line(header,1,'.Attr("epsilon: float")')
        write_line(header,1,'.Attr("steps: int")')
        write_line(header,1,'.Attr("numthreads: int")')
        write_line(header,1,'.Input("a_matrix: float")')
        write_line(header,1,'.Input("b_matrix: float")')
        write_line(header,1,'.Output("fast_mat_mul: float")')
        write_line(header,1,'.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {')
        write_line(header,2,'shape_inference::ShapeHandle A_shape;')
        write_line(header,2,'TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &A_shape)); // get shape of input matrix A')
        write_break(header)

        write_line(header,2,'shape_inference::ShapeHandle B_shape;')
        write_line(header,2,'TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &B_shape)); // get shape of input matrix B')
        write_break(header)

        write_line(header,2,'// get dimensions for the input matrices')
        write_line(header,2,'shape_inference::DimensionHandle A_rows = c->Dim(A_shape, 0);')
        write_line(header,2,'shape_inference::DimensionHandle B_cols = c->Dim(B_shape, 1);')
        write_break(header)

        write_line(header,2,'// set output matrix')
        write_line(header,2,'c->set_output(0, c->Matrix(A_rows, B_cols));')
        write_line(header,2,'return Status::OK();')
        write_line(header,1,'});')
        write_break(header)

        write_line(header,0,'class FastMatMulOp : public OpKernel {')
        write_line(header,0,'public:')
        write_line(header,1,'/// brief Constructor.')
        write_line(header,1,'/// param context')
        write_line(header,1,'explicit FastMatMulOp(OpKernelConstruction* context) : OpKernel(context) {')
        write_line(header,2,'// get attrs')
        write_break(header)

        write_line(header,2,'OP_REQUIRES_OK(context, context->GetAttr("steps", &numsteps_));')
        write_line(header,2,'OP_REQUIRES_OK(context,context->GetAttr("epsilon", &epsilon_));')
        write_line(header,2,'OP_REQUIRES_OK(context,context->GetAttr("numthreads", &numthreads_));')
        write_line(header,1,'}')
        write_break(header)

        write_line(header,1,'void Compute(OpKernelContext* context) override {')
        write_break(header)

        write_line(header,2,'// get the inputs')
        write_line(header,2,'const Tensor& A_matrix = context->input(0);')
        write_line(header,2,'const Tensor& B_matrix = context->input(1);')
        write_break(header)

        write_line(header,2,'// check shapes of inputs')
        write_line(header,2,'const TensorShape& A_shape = A_matrix.shape();')
        write_line(header,2,'const TensorShape& B_shape = B_matrix.shape();')
        write_break(header)

        write_line(header,2,'// check that they\'re matrices')
        write_line(header,2,'DCHECK_EQ(A_shape.dims(), 2);')
        write_line(header,2,'DCHECK_EQ(B_shape.dims(), 2);')
        write_break(header)

        write_line(header,2,'// check they can multiply')
        write_line(header,2,'DCHECK_EQ(A_shape.dim_size(1), B_shape.dim_size(0));')
        write_break(header)

        write_line(header,2,'// create output shape')
        write_line(header,2,'TensorShape output_shape;')
        write_line(header,2,'output_shape.AddDim(A_shape.dim_size(0));')
        write_line(header,2,'output_shape.AddDim(B_shape.dim_size(1));')
        write_break(header)

        write_line(header,2,'// create output tensor')
        write_line(header,2,'Tensor* output = NULL;')
        write_line(header,2,'OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));')
        write_break(header)

        write_line(header,2,'// cast tensor data as Matrix (class defined by arbenson)')
        write_line(header,2,'const float* a = static_cast<const float *>(DMAHelper::base(&A_matrix));')
        write_line(header,2,'const float* b = static_cast<const float *>(DMAHelper::base(&B_matrix));')
        write_line(header,2,'float* c = static_cast<float *>(DMAHelper::base(output));')
        write_line(header,2,'Matrix<float> A = Matrix<float>(b, B_shape.dim_size(1), B_shape.dim_size(1), B_shape.dim_size(0));')
        write_line(header,2,'Matrix<float> B = Matrix<float>(a, A_shape.dim_size(1), A_shape.dim_size(1), A_shape.dim_size(0));')
        write_line(header,2,'Matrix<float> C = Matrix<float>(c, output->dim_size(1), output->dim_size(1), output->dim_size(0));')
        write_break(header)

        write_line(header,2,'// call %s\'s matmul' % algo_name)
        write_line(header,2,'%s::FastMatmul(A, B, C, numsteps_, epsilon_, numthreads_);' % namespace_name)
        write_break(header)

        write_line(header,0,'//    const float* ptr = reinterpret_cast<const float*>(output->tensor_data().data());')
        write_line(header,0,'//    std::cout<< ptr[0] <<std::endl;')
        write_break(header)

        write_line(header,0,'//    float *ptr2 = static_cast<float *>(DMAHelper::base(output));')
        write_line(header,0,'//    ptr2[0] = 7;')
        write_line(header,0,'//    std::cout<< ptr2[0] <<std::endl;')
        write_break(header)

        write_line(header,1,'}')
        write_break(header)

        write_line(header,1,'private:')
        write_line(header,2,'int numsteps_;')
        write_line(header,2,'float epsilon_;')
        write_line(header,2,'int numthreads_;')
        write_break(header)

        write_line(header,1,'};')
        write_break(header)

        write_line(header,0,'REGISTER_KERNEL_BUILDER(Name("FastMatMul").Device(DEVICE_CPU), FastMatMulOp);')


if __name__ == "__main__":

    in_files = ['bini322_10_52_approx.hpp', "schonhage333_21_117_approx.hpp", "smirnov224_13_91_approx.hpp",
                "smirnov225_16_124_approx.hpp", "smirnov272_22_198_approx.hpp", "smirnov323_14_108_approx.hpp",
                "smirnov333_20_182_approx.hpp", "smirnov334_27_202_approx.hpp", "smirnov442_24_180_approx.hpp",
                "smirnov444_46_352_approx.hpp", "smirnov552_37_262_approx.hpp", "smirnov555_90_710_approx.hpp",
                "strassen.hpp"]

    for i in in_files:
        if i == "strassen.hpp":
            outfile = 'strassen_mat_mul.cc'
        else:
            outfile = f'src/{i.split("_")[0]}_mat_mul.cc'

        writer(in_file=i, out_file=outfile)



