// test program for tensor.h

#include "../tensor/tensor.h"
#include <chrono>
#include <cmath>

int main() {
  using namespace ssrn;
  using namespace std::chrono;
  using itr_t = tensor<float, 1>::index_t;

  int ret_code = 0;

  // 1. create tensor instance
  tensor<float, 2> A(4, 3); // declare 2d tensor, where 1st dim is 4, zero dim is 3.
  tensor<float, 2> B;       // declare 2d tensor, size is not defined here
  B.reshape(3, 4);          // reshape to 3*4

  if (A.num_elements() == B.num_elements()) {
    std::cout << "A and B has same number of elements : " << A.num_elements() << std::endl;
  } else {
    std::cout << "A and B has different size of elements." << std::endl;
    std::cout << "A has : " << A.num_elements() << std::endl;
    std::cout << "B has : " << B.num_elements() << std::endl;
    ret_code = 1;
  }

  // 2. calculate gemm

  // init data
  for (itr_t i = 0; i < A.num_elements(); i++) {
    A.data()[i] = i;
  }

  for (itr_t i = 0; i < B.shape<1>(); i++)
    for (itr_t j = 0; j < B.shape<0>(); j++) {
      B[i][j] = i * B.shape<0>() + j;
    }

  auto C = make_tensor<float, 2>(4, 4);
  std::fill(C.begin(), C.end(), 0);
  std::cout << "C has : " << C.num_elements() << std::endl;

  // compute gemm
  for (itr_t i = 0; i < A.shape<1>(); i++)
    for (itr_t j = 0; j < A.shape<1>(); j++)
      for (itr_t k = 0; k < A.shape<0>(); k++)
        C[i][j] += A[i][k] * B[k][j];

  const auto accum = std::accumulate(C.begin(), C.end(), 0);
  if (accum == 1580) {
    std::cout << "gemm calculation success : 1580" << std::endl;
    ;
  } else {
    std::cout << "gemm calculation failed : " << accum << std::endl;
    ret_code = 1;
  }

  int c_sum_indices = 0;
  for (itr_t i = 0; i < C.shape<1>(); i++)
    for (itr_t j = 0; j < C.shape<0>(); j++)
      c_sum_indices += C.with_indices(i, j);

  if (accum == c_sum_indices) {
    std::cout << "indice access success." << std::endl;
  } else {
    std::cout << "indice access failed." << std::endl;
    ret_code = 1;
  }

  // check util functions
  tensor<float, 3> D(4, 3, 2);
  std::iota(std::begin(D), std::end(D), 0);

  // make_view<N>(d1,d2,...)
  // get N dimension tensor_view from tensor.
  // in this case, you can get 2d tensor D_view from D(3d tensor).
  // template argument is the dimension of tensor_view you want to create
  // and function argumets are the index of upper dimension of D.
  // for example if D is 4*3*2 tensor and you call D.make_view<2>(1),
  // it returns 2d tensor_view, which references D[1][*][*].
  const auto& D_view = D.make_view<2>(1);
  std::cout << "D_view dim : " << D_view.dimension() << std::endl;

  auto E = D_view.to_tensor();
  std::cout << E.dimension() << std::endl;

  auto X = tensor<float, 2>(5, 3);
  auto Z = std::move(X);
  std::cout << Z.data() << std::endl;

  tensor<float, 2> Y;
  Y = std::move(Z);
  std::cout << Y.data() << std::endl;

  tensor<float, 2> W = std::move(Y);
  std::cout << Y.data() << std::endl;
  std::cout << W.data() << std::endl;

  return ret_code;
}
