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

  std::cout << std::numeric_limits<std::size_t>::max() << std::endl;
  std::cout << std::numeric_limits<long>::max() << std::endl;
  std::cout << 1 * 256 * 256 * 512L << std::endl;

  const int N = 51;

  tensor<float, 2, unsigned int> A, B, C;
  A.reshape(N, N);
  B.reshape({N, N});
  C.reshape({N, N});

  std::iota(A.begin(), A.end(), 1);
  std::iota(B.begin(), B.end(), 1);
  C.fill(0.f);

  auto st = std::chrono::system_clock::now();
  using itr_t = decltype(A)::index;
  for (itr_t i = 0; i < A.shape(2); i++)
    for (itr_t k = 0; k < A.shape(1); k++)
      for (itr_t j = 0; j < A.shape(1); j++)
        C.with_indices(i, j) += A.with_indices(i, k) * B.with_indices(k, j);
  // C[i][j] += A[i][k] * B[k][j];
  auto end = std::chrono::system_clock::now();

  auto sum = std::accumulate(C.begin(), C.end(), 0.0);
  std::cout << sum << std::endl;
  std::cout << duration_cast<milliseconds>(end - st).count() / 1000.f << std::endl;

  std::cout << sizeof(tensor<float, 2, std::size_t>::itype) << std::endl;
  std::cout << sizeof(tensor<float, 2, int>) << std::endl;
  std::cout << sizeof(tensor<float, 2>) << std::endl;

  const auto& a = A.make_view<1>({0});
  std::cout << a(1) << std::endl;

  tensor<float, 4> G{4, 3, 2, 1};
  std::iota(std::begin(G), std::end(G), 1);

  auto H = tensor_cast<3>(G);
  H.as_shape_of({4, 3, 2});

  tensor<float, 4> tt;
  tt.reshape(4, 3, 2, 1);
  std::cout << tt.shape(4) << std::endl;
}
