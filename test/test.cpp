#include "tensor.h"
#include <chrono>
#include <cmath>

int main(int argc, char* argv[]) {
  using namespace rnz;
  using namespace std::chrono;

  const int N = 4;
  const int NN = 2;
  tensor<float, 3> A, B, C, J;
  A.reshape(NN, N, N);
  B.reshape({NN, N, N});
  C.reshape({NN, N, N});

  std::iota(A.begin(), A.end(), 1);
  std::iota(B.begin(), B.end(), 1);
  C.fill(0.f);

  tensor<float, 1> K(2);
  tensor<float, 1> L(2);

  std::iota(K.begin(), K.end(), 1);
  std::iota(L.begin(), L.end(), 1);

  auto st = system_clock::now();
  using itr_t = decltype(A)::index_t;

  std::cout << "A shape 3: " << A.shape<3>() << std::endl;
  std::cout << "A shape 2: " << A.shape<2>() << std::endl;
  std::cout << "A shape 1: " << A.shape<1>() << std::endl;

  for (itr_t n = 0; n < A.shape<3>(); n++)
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (itr_t i = 0; i < A.shape<2>(); i++)
      for (itr_t k = 0; k < A.shape<1>(); k++) {
        for (itr_t j = 0; j < A.shape<1>(); j++) {
          C(n, i, j) += A(n, i, k) * B(n, k, j);
          // C.with_indices({n, i, j}) += A.with_indices({n, i, k}) * B.with_indices({n, k, j});
          /*
          C.data()[n * C.strides<2>() + i * C.strides<1>() + j] +=
              A.data()[n * A.strides<2>() + i * A.strides<1>() + k] *
              B.data()[n * B.strides<2>() + k * B.strides<1>() + j];
              */
        }
      }
  J.reshape(3, 2, 2);
  A.reshape(3, 2, 2);
  B.reshape(3, 2, 2);
  std::iota(A.begin(), A.end(), 1);
  std::iota(B.begin(), B.end(), 1);
  // tensor<float, 3> tar = A + B * 2;

  tensor<float, 1> bb = K + L;
  std::cout << bb[0] << std::endl;

  auto end = system_clock::now();

  auto sum = std::accumulate(C.begin(), C.end(), 0.0);
  std::cout << sum << std::endl;
  std::cout << duration_cast<milliseconds>(end - st).count() / 1000.f << std::endl;

  const auto& a = A.make_view<1>(0, 1);
  std::cout << "dimeinstion: " << a.dimension() << std::endl;
  for (auto e : a)
    std::cout << "e: " << e << std::endl;

  auto aa = a.to_tensor();

  tensor<float, 4> G{4, 3, 2, 1};
  std::iota(std::begin(G), std::end(G), 1);

  auto H = tensor_cast<3>(G);
  H.as_shape_of({4, 3, 2});

  tensor<float, 4> tt;
  tt.reshape(4, 3, 2, 1);
  std::iota(tt.begin(), tt.end(), 1);

  // auto tv = tt.make_view<3>(3);
}
