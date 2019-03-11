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

#include <chrono>
#include "tensor.h"

int main() {
  using namespace rnz;
  const int N = 512;

  // 2次元テンソルの生成: float A[N][N]と等価
  // 2次元テンソル = 行列　と考えてok
  // tensor<内部配列の型(float,double,int等), 次元>
  // 変数名({各次元のサイズをカンマ区切りしたもの})
  tensor<float, 2> A({N, N});
  tensor<float, 2> B({N, N});
  tensor<float, 2> C({N, N});

  // Cはゼロフィル
  std::fill(std::begin(C), std::end(C), 0.f);
  // AとBは1から始まる連番で埋める
  std::iota(std::begin(A), std::end(A), 1);
  std::iota(std::begin(B), std::end(B), 1);

  // 要素へのアクセス方法は３つある
  // 1つめの方法: 添字によるアクセス
  // 一番直感的だけど、一番遅いので速度が重視されない場面でどうぞ
  // shape(D): D次元目の要素数を返す
  for (int i = 0; i < A.shape(2); i++)
    for (int j = 0; j < A.shape(1); j++) C[i][j] = A[i][j] + B[i][j];

  std::fill(std::begin(C), std::end(C), 0.f);
  // 2つ目の方法
  // with_indicesメソッド経由でのアクセス
  // そこそこわかりやすい上に、下のポインタ経由でのアクセスとほぼ同等の速度
  // 個人的に一番おすすめ
  for (int i = 0; i < A.shape(2); i++)
    for (int j = 0; j < A.shape(1); j++)
      C.with_indices({i, j}) = A.with_indices({i, j}) + B.with_indices({i, j});

  std::fill(std::begin(C), std::end(C), 0.f);
  // 3つめの方法
  // dataメソッドで内部配列のポインタを取得してアクセス
  // わかりにくいけど、間違いなく最速

  /** stridesについて **
   *  三次元テンソル (三次元配列) A(dim3, dim2, dim1) を考えたときに
   *  A[i][j][k]へのアクセスを一次元的に表すと、
   *  A[i * dim2 * dim1 + j * dim1 + k]となる。
   *  このように、3次元目のインデックスである
   *  iの値が一つ変化すると、2次元目までの要素の合計である
   *  dim2 *dim1だけ位置がずれる。
   *  同じように、2次元目のインデックスである
   *  jの値が一つ変化すると, 1次元目の要素数dim1だけ位置がずれる。
   *  つまりD次元目を指すインデックスが変化すると、1次元目からD-1次元目までの要素の合計分だけ位置のずれが発生する。
   *  tensorでは、このずれをstrideとして保存しているおり、
   *  D次元目のインデックスに対応するずれをstrides(D-1)の形で取得できる
   */
  for (int i = 0; i < A.shape(2); i++)
    for (int j = 0; j < A.shape(1); j++)
      C.data()[i * C.strides(1) + j] =
          A.data()[i * A.strides(1) + j] + B.data()[i * B.strides(1) + j];

  // 内部配列の型と次元が同じならば、コピー可能
  tensor<float, 2> D = C;  // ok
  // tensor<double, 2>  D = C; // error
  // tensor<double, 3>  D = C; // error

  // 形状の変更
  // 次元の変更は不可
  // 内部配列が再確保されるので、保存されていたデータは消えます
  D.reshape({N * 3, N * 2});

  std::cout << D.shape(2) << std::endl;  // N * 3
  std::cout << D.shape(1) << std::endl;  // N * 2
}