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

int main(int argc, char *argv[]) {
  using namespace rnz;
  using namespace std::chrono;
  const int N = std::atoi(argv[1]);

  /*

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
  // あと、この方法はopenmpとの相性が良くない
  // shape(D): D次元目の要素数を返す
  auto st1 = system_clock::now();
  {
#pragma omp parallel for firstprivate(A, B)
    for (int i = 0; i < A.shape(2); i++) {
      for (int k = 0; k < A.shape(1); k++) {
        for (int j = 0; j < A.shape(2); j++) {
          C[i][j] += A[i][k] * B[k][j];
        }
      }
    }
  }
  auto end1 = system_clock::now();
  auto time1 = duration_cast<milliseconds>(end1 - st1).count() / 1000.f;

  auto sum1 = std::accumulate(std::begin(C), std::end(C), (double)0);
  std::fill(std::begin(C), std::end(C), 0.f);
  // 2つ目の方法
  // with_indicesメソッド経由でのアクセス
  // そこそこわかりやすい上に、下のポインタ経由でのアクセスとほぼ同等の速度
  // 個人的に一番おすすめ
  auto st2 = system_clock::now();
  {
#pragma omp parallel for firstprivate(A, B)
    for (int i = 0; i < A.shape(2); i++)
      for (int k = 0; k < A.shape(1); k++)
        for (int j = 0; j < A.shape(2); j++)
          C.with_indices(i, j) += A.with_indices(i, k) * B.with_indices(k, j);
  }
  auto end2 = system_clock::now();
  auto time2 = duration_cast<milliseconds>(end2 - st2).count() / 1000.f;

  auto sum2 = std::accumulate(std::begin(C), std::end(C), (double)0);
  std::fill(std::begin(C), std::end(C), 0.f);

  */

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

  /*
  auto st3 = system_clock::now();
  {
#pragma omp parallel for firstprivate(A, B)
    for (int i = 0; i < A.shape(2); i++)
      for (int k = 0; k < A.shape(1); k++)
        for (int j = 0; j < A.shape(2); j++)
          C.data()[i * C.strides(1) + j] +=
              A.data()[i * A.strides(1) + k] * B.data()[k * B.strides(1) + j];
  }
  auto end3 = system_clock::now();
  auto time3 = duration_cast<milliseconds>(end3 - st3).count() / 1000.f;
  auto sum3 = std::accumulate(std::begin(C), std::end(C), (double)0);

  std::cout << "method1: time " << time1 << "s: sum " << sum1 << std::endl;
  std::cout << "method2: time " << time2 << "s: sum " << sum2 << std::endl;
  std::cout << "method3: time " << time3 << "s: sum " << sum3 << std::endl;

  // 内部配列の型と次元が同じならば、コピー可能
  tensor<float, 2> D = C; // ok
  // tensor<double, 2>  D = C; // error
  // tensor<double, 3>  D = C; // error

  // 形状の変更
  // 次元の変更は不可
  // 内部配列が再確保されるので、保存されていたデータは消えます
  D.reshape({N * 3, N * 2});

  std::cout << D.shape(2) << std::endl; // N * 3
  std::cout << D.shape(1) << std::endl; // N * 2

  // 4次元テンソルの宣言
  // 宣言時に各次元のサイズを指定しなくてもよい
  tensor<float, 4> E;
  // あとからreshapeでサイズ変更
  E.reshape({4, 3, 2, 1});
  std::iota(E.begin(), E.end(), 1);

  std::cout << "shape E 4: " << E.shape(4) << std::endl;
  std::cout << "shape E 3: " << E.shape(3) << std::endl;
  std::cout << "shape E 2: " << E.shape(2) << std::endl;
  std::cout << "shape E 1: " << E.shape(1) << std::endl;
  std::cout << "here!" << std::endl;
  for (int m = 0; m < E.shape(4); m++) {
    std::cout << "m" << std::endl;
    for (int i = 0; i < E.shape(3); i++)
      for (int j = 0; j < E.shape(2); j++)
        for (int k = 0; k < E.shape(1); k++)
          std::cout << E[m][i][j][k] << " ";
  }

  std::cout << std::endl;

  std::cout << "here" << std::endl;
  */
  tensor<float, 3> E({2, 2, 3});
  std::iota(E.begin(), E.end(), 1);

  decltype(E)::view<2> Esub;
  Esub = (E.make_view<2>({0}));

  std::cout << std::endl;
  auto new_Esub = Esub.to_tensor();
  Esub[0][0] = 10;
  for (int i = 0; i < Esub.shape(2); i++)
    for (int j = 0; j < Esub.shape(1); j++)
      std::cout << Esub[i][j] << std::endl;

  std::cout << "here" << std::endl;
  for (int i = 0; i < new_Esub.shape(2); i++)
    for (int j = 0; j < new_Esub.shape(1); j++)
      std::cout << new_Esub[i][j] << std::endl;

  std::cout << "Esubsub" << std::endl;
  decltype(Esub)::view<1> Esubsub = Esub.make_view<1>({0});
  for (int i = 0; i < Esubsub.shape(1); i++)
    std::cout << Esubsub[i] << std::endl;
}