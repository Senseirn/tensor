/*  a sample program of tensor
 *
 *  A header-only tensor library.
 *  written by Yuta Kambara.
 */

#include "../tensor/tensor.h"
#include <chrono>
#include <cmath>

int main(int argc, char* argv[]) {
  using namespace ts;
  using namespace std::chrono;

  const int N = 4;

  // declare 2D tensor whose elements are float
  // the size of each dimension is N
  tensor<float, 2> A(N, N);

  // we can change the size later
  tensor<float, 2> B, C;
  B.reshape(N, N);
  C.reshape(N, N);

  std::iota(std::begin(B), std::end(B), 1);
  std::iota(std::begin(A), std::end(A), 1);
  std::iota(std::begin(C), std::end(C), 1);

  auto st = system_clock::now();
  C -= A + B * C;
  auto end = system_clock::now();

  std::cout << duration_cast<milliseconds>(end - st).count() / 1000.f << std::endl;

  for (auto e : C)
    std::cout << e << std::endl;

  std::cout << "---" << std::endl;
  std::cout << ts::is_simd_enabled() << std::endl;
  std::cout << ts::is_assert_enabled() << std::endl;

  //  auto AA = (A.make_view<1>(0) = A.make_view<1>(1) * D * D).to_tensor();
}
