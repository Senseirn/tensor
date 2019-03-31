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

  const int N = 4;

  // float型の２次元テンソルを宣言
  // 各次元の要素数はN
  tensor<float, 2> A(N, N);

  // あとから各次元の要素数を変更可
  tensor<float, 2> B, C;
  B.reshape(N, N);
  C.reshape(N, N);

  std::iota(std::begin(B), std::end(B), 1);
  std::iota(std::begin(A), std::end(A), 1);
  std::iota(std::begin(C), std::end(C), 1);

  auto st = system_clock::now();
  C -= A + B;
  auto end = system_clock::now();

  std::cout << duration_cast<milliseconds>(end - st).count() / 1000.f << std::endl;

  for (auto e : C)
    std::cout << e << std::endl;

  std::cout << "---" << std::endl;
  std::cout << rnz::is_simd_enabled() << std::endl;
  std::cout << rnz::is_asserts_enabled() << std::endl;

  //  auto AA = (A.make_view<1>(0) = A.make_view<1>(1) * D * D).to_tensor();
}
