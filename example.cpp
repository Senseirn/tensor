/*  a sample program of tensor
 *
 *  ヘッダオンリーのテンソルライブラリ
 *  テンソルは、（情報系の世界では）多次元配列とほぼ同義
 *
 *  本ライブラリでは、
 *  D階 (D次元)のテンソルを組み込み配列と同じように扱うためのクラスを提供する。
 *  クラス内部で動的にメモリ確保を行うため、組み込みの多次元配列と異なり
 *  スタックサイズによる配列サイズの制限を受けない一方で、
 *  組み込み配列と同様に、すべての次元に渡って連続的なメモリ配置を持つため、
 *  理論的なパフォーマンスは、組み込み配列と同等のはず。
 *  また、この特徴のため、内部配列のアドレスをblas等の関数へ渡せばそのまま
 *  高速な計算が行える。
 */

#include "tensor.h"
#include <chrono>
#include <cmath>

int main(int argc, char* argv[]) {
  using namespace rnz;
  using namespace std::chrono;

  const int N = 1024 / 2;
  const int NN = 2;
  tensor<float, 3, unsigned int> A, B, C, J;
  A.reshape(NN, N, N);
  B.reshape({NN, N, N});
  C.reshape({NN, N, N});

  std::iota(A.begin(), A.end(), 1);
  std::iota(B.begin(), B.end(), 1);
  C.fill(0.f);

  tensor<float, 3> K(3, 2, 2);
  tensor<float, 3> L(3, 2, 2);

  std::iota(K.begin(), K.end(), 1);
  L.fill(0.f);

  tensor<float, 2> M(2, 2);
  M = K.make_view<2>(1) + 2;

  auto st = system_clock::now();
  using itr_t = decltype(A)::index_t;

  for (itr_t n = 0; n < A.shape<3>(); n++)
#pragma omp parallel for
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
  auto ttttt = 2 * A + B + J * 3;
  auto end = system_clock::now();

  auto sum = std::accumulate(C.begin(), C.end(), 0.0);
  std::cout << sum << std::endl;
  std::cout << duration_cast<milliseconds>(end - st).count() / 1000.f << std::endl;

  const auto& a = A.make_view<1>(0);

  tensor<float, 4> G{4, 3, 2, 1};
  std::iota(std::begin(G), std::end(G), 1);

  auto H = tensor_cast<3>(G);
  H.as_shape_of({4, 3, 2});

  tensor<float, 4> tt;
  tt.reshape(4, 3, 2, 1);
  std::iota(tt.begin(), tt.end(), 1);

  auto tv = tt.make_view<3>(3);
}
