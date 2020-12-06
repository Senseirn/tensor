#define IUTEST_USE_MAIN 1
#include "iutest/include/iutest.hpp"

#include "../tensor/tensor.h"

using namespace ts;

IUTEST(TensorBasicTest, tensor1dBasicTest) {
  tensor<int, 1> a(3);                      // 1x3 tensor
  std::iota(std::begin(a), std::end(a), 1); // a should be {1,2,3}
  auto b = a;                               // b should be {1,2,3}

  IUTEST_ASSERT_EQ(3, a.num_elements());
  IUTEST_ASSERT_EQ(3, b.num_elements());
  IUTEST_ASSERT_EQ(3, a.shape<0>());
  IUTEST_ASSERT_EQ(3, b.shape<0>());

  IUTEST_ASSERT_EQ(6, std::accumulate(std::begin(a), std::end(a), 0));
  IUTEST_ASSERT_EQ(6, std::accumulate(std::begin(b), std::end(b), 0));

  using itr_t = tensor<int, 1>::index_t;
  for (itr_t i = 0; i < a.num_elements(); i++) {
    IUTEST_ASSERT_TRUE(a.data()[i] == b.data()[i]);
    IUTEST_ASSERT_TRUE(a.with_indices({i}) == b.with_indices({i}));
    IUTEST_ASSERT_TRUE(a(i) == b(i));
  }
}