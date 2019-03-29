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

  const int N = 2;

  tensor<float, 2, unsigned int> A, B, C, J;
  A.reshape(N, N);
  B.reshape({N, N});
  J.reshape({N, N});
  C.reshape({N, N});

  std::iota(A.begin(), A.end(), 1);
  std::iota(B.begin(), B.end(), 1);
  std::iota(J.begin(), J.end(), 1);
  C.fill(0.f);

  /*
  using itr_t = decltype(A)::index_t;
  for (itr_t i = 0; i < A.shape(2); i++)
    for (itr_t k = 0; k < A.shape(1); k++)
      for (itr_t j = 0; j < A.shape(1); j++)
        C(i, j) += A(i, k) * B(k, j);
        */
  // C[i][j] += A[i][k] * B[k][j];
  // auto ttttt = A + B + J;
  C = A + (2 * (B * J) + 3);

  // std::cout << typeid(ttttt).name() << std::endl;
  using itr_t = decltype(C)::index_t;
  for (itr_t i = 0; i < C.shape(2); i++)
    for (itr_t k = 0; k < C.shape(1); k++)
      std::cout << C(i, k) << std::endl;

  auto sum = std::accumulate(C.begin(), C.end(), 0.0);
  std::cout << sum << std::endl;

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
