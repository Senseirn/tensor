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

  // float型の２次元テンソルを宣言
  // 各次元の要素数はN
  tensor<float, 2> A(N, N);

  // あとから各次元の要素数を変更可
  tensor<float, 2> B, C;
  B.reshape(N, N);
  C.reshape(N, N);

  tensor<float, 1> D(N);
  std::iota(std::begin(D), std::end(D), 1);
  std::iota(std::begin(A), std::end(A), 1);
  C.fill(0);

  auto AA = (A.make_view<1>(0) = A.make_view<1>(1) * D * D).to_tensor();

  for (auto e : AA) {
    std::cout << e << std::endl;
  }
}
