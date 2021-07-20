#define IUTEST_USE_MAIN 1
#include "iutest/include/iutest.hpp"

#include "../tensor/tensor.h"

using namespace ssrn;

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

  tensor<int, 1> c = a * 2;
  for (itr_t i = 0; i < a.num_elements(); i++) {
    IUTEST_ASSERT_TRUE(a.data()[i] * 2 == c.data()[i]);
    IUTEST_ASSERT_TRUE(a.with_indices({i}) * 2 == c.with_indices({i}));
    IUTEST_ASSERT_TRUE(a(i) * 2 == c(i));
  }

  a = a * 2;
  for (itr_t i = 0; i < a.num_elements(); i++) {
    IUTEST_ASSERT_TRUE(a.data()[i] == c.data()[i]);
    IUTEST_ASSERT_TRUE(a.with_indices({i}) == c.with_indices({i}));
    IUTEST_ASSERT_TRUE(a(i) == c(i));
  }

  // Test fill function
  tensor<int, 1> d;
  d.reshape(5).fill(5);
  IUTEST_ASSERT_EQ(5, d.num_elements());
  auto result1 = std::find_if_not(std::begin(d), std::end(d), [](int x) { return x == 5; }) == std::end(d);
  IUTEST_ASSERT_TRUE(result1 == true);

  tensor<int, 2> e;
  e.reshape(3, 3).fill(10);
  IUTEST_ASSERT_EQ(9, e.num_elements());
  auto result2 = std::find_if_not(std::begin(e), std::end(e), [](int x) { return x == 10; }) == std::end(e);
  IUTEST_ASSERT_TRUE(result2 == true);
}