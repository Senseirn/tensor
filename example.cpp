#include <chrono>
#include "tensor.h"

int main() {
  // 4次元テンソルの生成: float A[4][3][2][1]と等価
  const int N = 512 * 4;
  tensor<float, 2> A{N, N};
  tensor<float, 2> B{N, N};
  tensor<float, 2> C{N, N};

  tensor<float, 4> D{4, 3, 2, 1};
  // ゼロフィル
  C.fill(0.f);
  // 1から始まる連番で埋める
  std::iota(std::begin(A), std::end(A), 1.f);
  std::iota(std::begin(B), std::end(B), 1.f);
  std::iota(std::begin(D), std::end(D), 1.f);

  for (int i = 0; i < D.shape(4); i++)
    for (int j = 0; j < D.shape(3); j++)
      for (int k = 0; k < D.shape(2); k++)
        for (int l = 0; l < D.shape(1); l++)
          std::cout << D.with_indices({i, j, k, l}) << std::endl;

  // A.shape(n): n次元目の要素数を取得

  using namespace std::chrono;

  auto st = system_clock::now();
#pragma omp parallel for firstprivate(A, B)
  for (int i = 0; i < N; i++)
    for (int k = 0; k < N; k++)
      for (int j = 0; j < N; j++)
        C.with_indices({i, j}) +=
            A.with_indices({i, k}) * B.with_indices({k, j});
  // C[i][j] += A[i][k] * B[k][j];
  auto end = system_clock::now();
  auto time = duration_cast<milliseconds>(end - st).count() / 1000.f;
  std::cout << time << std::endl;
  std::cout << std::accumulate(std::begin(C), std::end(C), 0.0) << std::endl;
  std::cout << std::endl;

  C.fill(0.f);

  auto st2 = system_clock::now();
#pragma omp parallel for firstprivate(A, B)
  for (int i = 0; i < N; i++)
    for (int k = 0; k < N; k++)
      for (int j = 0; j < N; j++)
        C.data()[i * C.strides(1) + j] +=
            A.data()[i * A.strides(1) + k] * B.data()[k * B.strides(1) + j];
  auto en2 = system_clock::now();
  auto time2 = duration_cast<milliseconds>(en2 - st2).count() / 1000.f;
  std::cout << time2 << std::endl;
  std::cout << std::accumulate(std::begin(C), std::end(C), 0.0) << std::endl;
}