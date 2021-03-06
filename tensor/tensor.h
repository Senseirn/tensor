/**
 * @file tensor_core.h
 * @brief tensor core header
 * @author Yuta Kambara
 */

#pragma once

/*--- check if the compiler supports C++11 ---*/
#if __cplusplus < 201103L
#error Tensor library needs at least a C++11 compliant compiler
#endif

/*--- if TENSOR_ENABLE_ASSERTS macro is NOT defined ---*/
#if not defined(TENSOR_ENABLE_ASSERTS) && not defined(NDEBUG)
#define NDEBUG
#endif

/*--- if SIMD is enabled but the compiler does not support it ---*/
#if defined(TENSOR_ENABLE_SIMD)
#if not defined(__AVX__)
#error The compiler does not support SIMD extensions (add options like '-mavx' or '-march=native').
#endif
#endif

/*--- if AVX macro is defined ---*/
#if defined(__AVX__) && not defined(TENSOR_ENABLE_SIMD)
#define TENSOR_ENABLE_SIMD
#endif

/*--- define likely and unlikely macros ---*/

// if the compiler is GCC, Clang or ICC,
// we use builtin_expect intrinsics.
#if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif

/*--- define default internal type ---*/
#ifndef TENSOR_DEFAULT_INTERNAL_TYPE
#define TENSOR_DEFAULT_INTERNAL_TYPE int32_t
#endif

/*--- define name of namespace(default is ssrn) ---*/
#ifndef TENSOR_NAMESPACE_NAME
#define TENSOR_NAMESPACE_NAME ssrn
#endif

#include <algorithm>
#include <array>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <vector>

#ifdef TENSOR_ENABLE_SIMD
#include <immintrin.h>
#endif

namespace TENSOR_NAMESPACE_NAME {

/* the bese class of all tensor classes */
class tensor_internal {};

/*--- forward declarations ---*/
template <typename T, std::size_t D, typename INTERNAL_TYPE = TENSOR_DEFAULT_INTERNAL_TYPE>
struct tensor_extent;
template <typename T, std::size_t D, typename INTERNAL_TYPE = TENSOR_DEFAULT_INTERNAL_TYPE, typename = void>
class tensor;
template <typename T, std::size_t D, typename INTERNAL_TYPE = TENSOR_DEFAULT_INTERNAL_TYPE>
class tensor_view;

template <typename L, typename Op, typename R, typename T, typename ITYPE = TENSOR_DEFAULT_INTERNAL_TYPE>
class Expr;

/*--- range check functions ---*/
static inline constexpr bool is_assert_enabled() {
#ifdef TENSOR_ENABLE_ASSERTS
  return true;
#else
  return false;
#endif
}

template <
    typename T1,
    typename T2,
    typename std::enable_if<std::is_integral<T1>::value && std::is_integral<T2>::value, std::nullptr_t>::type = nullptr>
void check_range(const T1 i, const T2 dim) {
  if (is_assert_enabled())
    if (unlikely(i < 0 || i >= dim)) {
      std::cerr << "error: out of range access (idx=" << i << ", dim=" << dim << ")" << std::endl;
      std::exit(1);
    }
}

template <
    typename T1,
    typename T2,
    typename T3,
    typename std::enable_if<std::is_integral<T1>::value && std::is_integral<T2>::value && std::is_integral<T3>::value,
                            std::nullptr_t>::type = nullptr>
void check_range(const T1 i, const T2 dim, const T3 D) {
  if (is_assert_enabled())
    if (unlikely(i < 0 || i >= dim)) {
      std::cerr << "error: out of range access [trying to access " << i << "th element in " << D
                << "th dimension(max range = " << dim - 1 << ")]" << std::endl;
      std::exit(1);
    }
}

template <std::size_t D1, typename T, std::size_t D2, typename INTERNAL_TYPE>
tensor<T, D1, INTERNAL_TYPE> tensor_cast(const tensor<T, D2, INTERNAL_TYPE>& src) {
  tensor<T, D1, INTERNAL_TYPE> retval;
  retval.reshape(src.num_elements());
  std::copy(std::begin(src), std::end(src), std::begin(retval));
  return retval;
}

static inline constexpr bool is_simd_enabled() {
#ifdef TENSOR_ENABLE_SIMD
  return true;
#else
  return false;
#endif
}

template <typename T, std::size_t Align = 32>
T* aligned_alloc(std::size_t n) {
#ifdef TENSOR_ENABLE_SIMD
  return static_cast<T*>(_mm_malloc(sizeof(T) * n, 32));
#else
  return new T[n];
#endif
}

template <typename T>
void aligned_deleter(T* p) {
#ifdef TENSOR_ENABLE_SIMD
  _mm_free(p);
#else
  delete[] p;
#endif
}

/*--- functions ---*/

// a helper function returning a tensor object.
/*
template <typename T, std::size_t D, typename U, typename INTERNAL_TYPE = TENSOR_DEFAULT_INTERNAL_TYPE>
tensor<T, D, INTERNAL_TYPE> make_tensor(const std::initializer_list<U>& initialzier) {
  return tensor<T, D, INTERNAL_TYPE>(initialzier);
}
*/

template <typename T, std::size_t D, typename INTERNAL_TYPE = TENSOR_DEFAULT_INTERNAL_TYPE, typename... Args>
tensor<T, D, INTERNAL_TYPE> make_tensor(Args... args) {
  static_assert(sizeof...(args) == D, "error: dimension miss-match");
  return tensor<T, D, INTERNAL_TYPE>(args...);
}

/*--- class and struct ---*/
template <typename T, std::size_t D, typename INTERNAL_TYPE>
class tensor_view : public tensor_internal {
 public:
  typedef INTERNAL_TYPE _internal_t;

 private:
  T* _data; // a pointer to the first data of dimension D
  _internal_t _num_elements;
  std::vector<_internal_t> _dims;    // the num of elements in each dimension
  std::vector<_internal_t> _strides; // access strides
  tensor_extent<T, D - 1, _internal_t> _extents;

  T eval(const int i) const { return _data[i]; }

 public:
  /*--- typedefs ---*/
  typedef T value_type;
  typedef std::array<_internal_t, D> multi_index;
  typedef _internal_t index_t;

  template <std::size_t _D>
  using view = tensor_view<T, _D, _internal_t>;

  template <std::size_t _D>
  using fixed_indices = std::array<int, _D>;

  template <typename L, typename Op, typename R, typename TT, typename ITYPE>
  friend class Expr;

  /*--- constructors ---*/
  tensor_view()
  : _data(nullptr)
  , _dims(D)
  , _strides(D) {}
  tensor_view(T* p, const std::vector<_internal_t>& dims, const std::vector<_internal_t>& strides)
  : _data(p)
  , _dims(D)
  , _strides(D) {
    const auto size_diff = dims.size() - D;
    std::copy(std::begin(dims), std::end(dims) - size_diff, std::begin(_dims));
    std::copy(std::begin(strides), std::end(strides) - size_diff, std::begin(_strides));
    _num_elements = std::accumulate(std::begin(_dims), std::end(_dims), 1, std::multiplies<int>());
    _extents.init(_data, _dims, _strides);
  }

  tensor_view(const tensor_view& src)
  : _data(src.data())
  , _num_elements(src.num_elements()) {
    _dims = src.dims();
    _strides = src.strides();
    _extents.init(_data, _dims, _strides);
  }

  tensor_view(tensor_view&& src)
  : _data(src.data())
  , _num_elements(src.num_elements()) {
    _dims = std::move(src.dims());
    _strides = std::move(src.strides());
    _extents.init(_data, _dims, _strides);
  }

  /*--- operators ---*/
  tensor_view& operator=(const tensor_view& src) {
    _data = src.data();
    _num_elements = src.num_elements();
    _dims = src.dims();
    _strides = src.strides();
    _extents.init(_data, _dims, _strides);
    return *this;
  }

  tensor_view& operator=(tensor_view&& src) {
    _data = src.data();
    _num_elements = src.num_elements();
    _dims = std::move(src.dims());
    _strides = std::move(src.strides());
    _extents.init(_data, _dims, _strides);
    return *this;
  }

  template <typename L, typename Op, typename R, typename ITYPE>
  tensor_view& operator=(const Expr<L, Op, R, value_type, ITYPE>& rhs) {
    static_assert(std::is_same<_internal_t, ITYPE>::value, "tensor assignment error: different internal type");
    if (rhs.num_elements() != num_elements()) {
      std::cerr << "tensor assignment error: different num of elements" << std::endl;
    }
    for (auto i = to_index(0); i < num_elements(); i++) {
      _data[i] = rhs.eval(i);
    }
    return *this;
  }

  template <typename L, typename Op, typename R, typename ITYPE>
  tensor_view& operator+=(const Expr<L, Op, R, value_type, ITYPE>& rhs) {
    static_assert(std::is_same<_internal_t, ITYPE>::value, "tensor assignment error: different internal type");
    if (rhs.num_elements() != num_elements()) {
      std::cerr << "tensor assignment error: different num of elements" << std::endl;
    }
    for (auto i = to_index(0); i < num_elements(); i++) {
      _data[i] += rhs.eval(i);
    }
    return *this;
  }

  template <typename L, typename Op, typename R, typename ITYPE>
  tensor_view& operator-=(const Expr<L, Op, R, value_type, ITYPE>& rhs) {
    static_assert(std::is_same<_internal_t, ITYPE>::value, "tensor assignment error: different internal type");
    if (rhs.num_elements() != num_elements()) {
      std::cerr << "tensor assignment error: different num of elements" << std::endl;
    }
    for (auto i = to_index(0); i < num_elements(); i++) {
      _data[i] -= rhs.eval(i);
    }
    return *this;
  }

  template <typename L, typename Op, typename R, typename ITYPE>
  tensor_view& operator/=(const Expr<L, Op, R, value_type, ITYPE>& rhs) {
    static_assert(std::is_same<_internal_t, ITYPE>::value, "tensor assignment error: different internal type");
    if (rhs.num_elements() != num_elements()) {
      std::cerr << "tensor assignment error: different num of elements" << std::endl;
    }
    for (auto i = to_index(0); i < num_elements(); i++) {
      _data[i] /= rhs.eval(i);
    }
    return *this;
  }

  template <typename L, typename Op, typename R, typename ITYPE>
  tensor_view& operator*=(const Expr<L, Op, R, value_type, ITYPE>& rhs) {
    static_assert(std::is_same<_internal_t, ITYPE>::value, "tensor assignment error: different internal type");
    if (rhs.num_elements() != num_elements()) {
      std::cerr << "tensor assignment error: different num of elements" << std::endl;
    }
    for (auto i = to_index(0); i < num_elements(); i++) {
      _data[i] *= rhs.eval(i);
    }
    return *this;
  }

  tensor_extent<T, D - 1, _internal_t>& operator[](const _internal_t i) {
    check_range(i, _dims[D - 1], D);
    return _extents.calc_index(i * _strides[D - 1]);
  }

  const tensor_extent<T, D - 1, _internal_t>& operator[](const _internal_t i) const {
    check_range(i, _dims[D - 1], D);
    return _extents.calc_index(i * _strides[D - 1]);
  }

  /*--- member functions ---*/
  T* begin() { return _data; }

  T* end() { return _data + _num_elements; }

  T* begin() const { return _data; }

  T* end() const { return _data + _num_elements; }

  T*& data() { return _data; }

  T* const& data() const { return _data; }

  _internal_t num_elements() const { return _num_elements; }

  _internal_t shape() const { return D; }

  _internal_t shape(const _internal_t d) const { return _dims[d]; }

  template <std::size_t _D, typename std::enable_if<(_D <= D && _D >= 0), std::nullptr_t>::type = nullptr>
  _internal_t shape() {
    return _dims[_D];
  }

  template <std::size_t _D, typename std::enable_if<(_D <= D && _D >= 0), std::nullptr_t>::type = nullptr>
  _internal_t shape() const {
    return _dims[_D];
  }

  const std::vector<_internal_t>& dims() const { return _dims; }

  _internal_t strides(const _internal_t d) const { return _strides[d]; }

  const std::vector<_internal_t>& strides() const { return _strides; }

  _internal_t dimension() const { return (_internal_t)D; }

  T& with_indices(const multi_index& indices) {
    for (int i = D - 1; i >= 0; --i) {
      check_range(indices[D - 1 - i], _dims[i], i + 1);
    }
    _internal_t idx = 0;
    for (int i = D - 1; i > 0; --i) {
      idx += indices[D - 1 - i] * strides(i);
    }
    idx += indices[D - 1];
    return _data[idx];
  }

  T with_indices(const multi_index& indices) const {
    for (int i = D - 1; i >= 0; --i) {
      check_range(indices[D - 1 - i], _dims[i], i + 1);
    }
    _internal_t idx = 0;
    for (int i = D - 1; i > 0; --i) {
      idx += indices[D - 1 - i] * strides(i);
    }
    idx += indices[D - 1];
    return _data[idx];
  }

  template <typename... Args>
  T& with_indices(Args... args) {
    return with_indices({args...});
  }

  template <typename... Args>
  T with_indices(Args... args) const {
    return with_indices({args...});
  }

  template <typename... Args>
  T& operator()(Args... args) {
    static_assert(sizeof...(args) == D, "error: dimension miss-match");
    return with_indices({static_cast<_internal_t>(args)...});
  }

  template <typename... Args>
  T operator()(Args... args) const {
    static_assert(sizeof...(args) == D, "error: dimension miss-match");
    return with_indices({static_cast<_internal_t>(args)...});
  }

  tensor<T, D, _internal_t> to_tensor() const { return tensor<T, D, _internal_t>(*this); }

  template <std::size_t _D>
  view<_D> make_view(const fixed_indices<D - _D>& indices) {
    static_assert(D - _D > 0, "dimension of view must be greater than 0.");
    multi_index midx;
    std::fill(midx.begin(), midx.end(), 0);
    std::copy(indices.begin(), indices.end(), midx.begin());
    return view<_D>(&with_indices(midx), _dims, _strides);
  }

  template <std::size_t _D, typename... Args>
  view<_D> make_view(Args... args) {
    return make_view<_D>({static_cast<_internal_t>(args)...});
  }

  template <typename type>
  _internal_t to_index(const type i) {
    return static_cast<_internal_t>(i);
  }

  template <typename type>
  tensor_view& fill(type x) {
    std::fill(_data, _data + _num_elements, (T)x);
    return *this;
  }
};

template <typename T, typename INTERNAL_TYPE>
class tensor_view<T, 1, INTERNAL_TYPE> : public tensor_internal {
 public:
  typedef INTERNAL_TYPE _internal_t;

 private:
  T* _data;                          // a pointer to the data
  _internal_t _num_elements;         // total elements of tensor
  std::vector<_internal_t> _dims;    // the num of elements in each dimension
  std::vector<_internal_t> _strides; // access strides
  // tensor_extent<T, D - 1> _extents;  // inner struct to calculate index
  T eval(const int i) const { return _data[i]; }

 public:
  /*--- typedefs ---*/
  typedef T value_type;
  typedef _internal_t multi_index;
  typedef _internal_t index_t;

  template <typename L, typename Op, typename R, typename TT, typename ITYPE>
  friend class Expr;
  /*--- constructors ---*/
  tensor_view()
  : _data(nullptr)
  , _num_elements(0)
  , _dims(1)
  , _strides(1) {}

  // constructor: acceptes initililzer_list whose size is D
  tensor_view(T* p, const std::vector<_internal_t>& dims, const std::vector<_internal_t>& strides)
  : _data(p)
  , _dims(1)
  , _strides(1) {
    // std::reverse(_dims.begin(), _dims.end());
    _num_elements = dims[0];
    _dims[0] = _num_elements;
    _strides[0] = 1;
    (void)strides;
  }

  tensor_view(const tensor_view& src)
  : _data(src.data())
  , _num_elements(src.num_elements()) {
    _dims = src.dims();
    _strides = src.strides();
  }

  tensor_view(tensor_view&& src)
  : _data(src.data())
  , _num_elements(src.num_elements()) {
    _dims = std::move(src.dims());
    _strides = std::move(src.strides());
  }

  /*--- operators ---*/
  tensor_view& operator=(const tensor_view& src) {
    _data = src.data();
    _num_elements = src.num_elements();
    _dims = src.dims();
    _strides = src.strides();
    return *this;
  }

  tensor_view& operator=(tensor_view&& src) {
    _data = src.data();
    _num_elements = src.num_elements();
    _dims = std::move(src.dims());
    _strides = std::move(src.strides());
    return *this;
  }

  template <typename L, typename Op, typename R, typename ITYPE>
  tensor_view& operator=(const Expr<L, Op, R, value_type, ITYPE>& rhs) {
    static_assert(std::is_same<_internal_t, ITYPE>::value, "tensor assignment error: different internal type");
    if (rhs.num_elements() != num_elements()) {
      std::cerr << "tensor assignment error: different num of elements" << std::endl;
    }
    for (auto i = to_index(0); i < num_elements(); i++) {
      _data[i] = rhs.eval(i);
    }
    return *this;
  }

  template <typename L, typename Op, typename R, typename ITYPE>
  tensor_view& operator+=(const Expr<L, Op, R, value_type, ITYPE>& rhs) {
    static_assert(std::is_same<_internal_t, ITYPE>::value, "tensor assignment error: different internal type");
    if (rhs.num_elements() != num_elements()) {
      std::cerr << "tensor assignment error: different num of elements" << std::endl;
    }
    for (auto i = to_index(0); i < num_elements(); i++) {
      _data[i] += rhs.eval(i);
    }
    return *this;
  }

  template <typename L, typename Op, typename R, typename ITYPE>
  tensor_view& operator-=(const Expr<L, Op, R, value_type, ITYPE>& rhs) {
    static_assert(std::is_same<_internal_t, ITYPE>::value, "tensor assignment error: different internal type");
    if (rhs.num_elements() != num_elements()) {
      std::cerr << "tensor assignment error: different num of elements" << std::endl;
    }
    for (auto i = to_index(0); i < num_elements(); i++) {
      _data[i] -= rhs.eval(i);
    }
    return *this;
  }

  template <typename L, typename Op, typename R, typename ITYPE>
  tensor_view& operator/=(const Expr<L, Op, R, value_type, ITYPE>& rhs) {
    static_assert(std::is_same<_internal_t, ITYPE>::value, "tensor assignment error: different internal type");
    if (rhs.num_elements() != num_elements()) {
      std::cerr << "tensor assignment error: different num of elements" << std::endl;
    }
    for (auto i = to_index(0); i < num_elements(); i++) {
      _data[i] /= rhs.eval(i);
    }
    return *this;
  }

  template <typename L, typename Op, typename R, typename ITYPE>
  tensor_view& operator*=(const Expr<L, Op, R, value_type, ITYPE>& rhs) {
    static_assert(std::is_same<_internal_t, ITYPE>::value, "tensor assignment error: different internal type");
    if (rhs.num_elements() != num_elements()) {
      std::cerr << "tensor assignment error: different num of elements" << std::endl;
    }
    for (auto i = to_index(0); i < num_elements(); i++) {
      _data[i] *= rhs.eval(i);
    }
    return *this;
  }

  T& operator[](const _internal_t i) {
    check_range(i, _dims[0], 1);
    return _data[i];
  }

  const T operator[](const _internal_t i) const {
    check_range(i, _dims[0], 1);
    return _data[i];
  }

  /*--- member functions ---*/
  T* begin() { return _data; }

  T* end() { return _data + _num_elements; }

  T* begin() const { return _data; }

  T* end() const { return _data + _num_elements; }

  T*& data() { return _data; }

  T* data() const { return _data; }

  _internal_t num_elements() const { return _num_elements; }

  _internal_t shape() const { return 1; }

  _internal_t shape(const _internal_t d) const { return _dims[d]; }

  template <std::size_t _D, typename std::enable_if<(_D == 0), std::nullptr_t>::type = nullptr>
  _internal_t shape() {
    return _dims[_D];
  }

  template <std::size_t _D, typename std::enable_if<(_D == 0), std::nullptr_t>::type = nullptr>
  _internal_t shape() const {
    return _dims[_D];
  }

  const std::vector<_internal_t>& dims() const { return _dims; }

  _internal_t strides(const _internal_t d) const { return _strides[d]; }

  const std::vector<_internal_t>& strides() const { return _strides; }

  _internal_t dimension() const { return (_internal_t)1; }

  T& with_indices(const _internal_t indices) {
    check_range(indices, _dims[0], 1);
    return _data[indices];
  }

  T with_indices(const _internal_t indices) const {
    check_range(indices, _dims[0], 1);
    return _data[indices];
  }

  T& operator()(const _internal_t indices) {
    check_range(indices, _dims[0], 1);
    return _data[indices];
  }

  T operator()(const _internal_t indices) const {
    check_range(indices, _dims[0], 1);
    return _data[indices];
  }

  tensor<T, 1, _internal_t> to_tensor() const { return tensor<T, 1, _internal_t>(*this); }

  template <typename type>
  _internal_t to_index(const type i) {
    return static_cast<_internal_t>(i);
  }

  template <typename type>
  tensor_view<T, 1, INTERNAL_TYPE>& fill(type x) {
    std::fill(_data, _data + _num_elements, (T)x);
    return *this;
  }
};

template <typename T, std::size_t D, typename INTERNAL_TYPE>
struct tensor_extent : public tensor_internal {
  typedef INTERNAL_TYPE _internal_t;
  T* _p;
  _internal_t _dim;    // D次元の要素数
  _internal_t _stride; // D-1次元までの要素数
  tensor_extent<T, D - 1, _internal_t> _extents;
  mutable _internal_t _accum;

  //  T eval(const int i) const { return _data[i]; }

  template <typename L, typename Op, typename R, typename TT, typename ITYPE>
  friend class Expr;
  /*--- functions ---*/
  inline tensor_extent<T, D, _internal_t>& calc_index(const _internal_t accum) {
    _accum = accum;
    return *this;
  }

  inline const tensor_extent<T, D, _internal_t>& calc_index(const _internal_t accum) const {
    _accum = accum;
    return *this;
  }

  tensor_extent()
  : _p(nullptr)
  , _stride(0)
  , _extents(nullptr, 0) {}

  tensor_extent(T* p, _internal_t stride)
  : _p(p)
  , _stride(stride)
  , _extents(p, stride) {}

  tensor_extent<T, D - 1, _internal_t>& operator[](const _internal_t i) {
    check_range(i, _dim, D);
    return _extents.calc_index(i * _stride + _accum);
  }

  const tensor_extent<T, D - 1, _internal_t>& operator[](const _internal_t i) const {
    check_range(i, _dim, D);
    return _extents.calc_index(i * _stride + _accum);
  }

  _internal_t accum() const { return _accum; }

  void init(T* p, std::vector<_internal_t>& dims, std::vector<_internal_t>& strides) {
    _p = p;
    _dim = dims[D - 1];
    _stride = strides[D - 1];
    _extents.init(p, dims, strides);
  }
};

template <typename T, typename INTERNAL_TYPE>
struct tensor_extent<T, 1, INTERNAL_TYPE> : public tensor_internal {
  typedef INTERNAL_TYPE _internal_t;
  T* _p;
  _internal_t _dim;
  mutable _internal_t _accum;
  //  T eval(const int i) const { return _data[i]; }

  /*--- functions ---*/
  tensor_extent()
  : _p(nullptr) {}
  tensor_extent(T* p, _internal_t stride)
  : _p(p) {
    (void)stride;
  }

  tensor_extent<T, 1, _internal_t>& calc_index(const _internal_t accum) {
    _accum = accum;
    return *this;
  }

  const tensor_extent<T, 1, _internal_t>& calc_index(const _internal_t accum) const {
    _accum = accum;
    return *this;
  }

  void init(T* p, std::vector<_internal_t>& dims, std::vector<_internal_t>& strides) {
    _dim = dims[0];
    _p = p;
    (void)strides;
  }

  inline T& operator[](const _internal_t i) {
    check_range(i, _dim, 1);
    return _p[i + _accum];
  }

  inline const T& operator[](const _internal_t i) const {
    check_range(i, _dim, 1);
    return _p[i + _accum];
  }
};

template <typename T, std::size_t D, typename INTERNAL_TYPE>
class tensor<T,
             D,
             INTERNAL_TYPE,
             typename std::enable_if<std::is_arithmetic<T>::value || std::is_pointer<T>::value, void>::type>
: public tensor_internal {
 public:
  typedef INTERNAL_TYPE _internal_t;

 private:
  T* _data;                                      // a pointer to the data
  _internal_t _num_elements;                     // total elements of tensor
  std::vector<_internal_t> _dims;                // the num of elements in each dimension
  std::vector<_internal_t> _strides;             // access strides
  tensor_extent<T, D - 1, _internal_t> _extents; // inner struct to calculate index
  T eval(const int i) const { return _data[i]; }

 public:
  /*--- public typedefs ---*/
  typedef T value_type;
  typedef std::array<_internal_t, D> multi_index;

  template <std::size_t _D>
  using view = tensor_view<T, _D, _internal_t>;

  template <std::size_t _D>
  using fixed_indices = std::array<_internal_t, _D>;

  typedef _internal_t index_t;

  template <typename L, typename Op, typename R, typename TT, typename ITYPE>
  friend class Expr;

  /*--- constructors ---*/
  tensor()
  : _data(nullptr)
  , _dims(D)
  , _strides(D) {}

  // constructor: acceptes initililzer_list whose size is D
  tensor(std::initializer_list<_internal_t> i_list)
  : _data(nullptr)
  , _dims(D)
  , _strides(D) {
    // error check
    // if num of argument is not same as D
    if (i_list.size() != D) {
      std::cerr << "error: dimension miss-match" << std::endl;
      std::exit(1);
    }
    std::copy(i_list.begin(), i_list.end(), _dims.begin());
    // if one of arguments is 0
    if (std::find(std::begin(_dims), std::end(_dims), 0) != std::end(_dims)) {
      std::cerr << "error: 0 is not permitted as a size of dimension" << std::endl;
      std::exit(1);
    }
    std::reverse(_dims.begin(), _dims.end());
    _num_elements = std::accumulate(_dims.begin(), _dims.end(), 1, std::multiplies<int>());
    //_data = new T[_num_elements];
    _data = aligned_alloc<T>(_num_elements);
    _strides[D - 1] = _num_elements / _dims[D - 1];
    for (int i = D - 2; i >= 0; i--) {
      _strides[i] = _strides[i + 1] / _dims[i];
    }
    _extents.init(_data, _dims, _strides);
  }

  template <typename... Args>
  tensor(Args... args)
  : tensor({static_cast<_internal_t>(args)...}) {}

  tensor(const tensor_view<T, D, _internal_t>& view)
  : _data(nullptr)
  , _dims(D)
  , _strides(D) {
    _num_elements = view.num_elements();
    _dims = view.dims();
    _strides = view.strides();
    //_data = new T[_num_elements];
    _data = aligned_alloc<T>(_num_elements);
    std::copy(view.begin(), view.end(), _data);
    _extents.init(_data, _dims, _strides);
  }

  tensor(const tensor& src) {
    _num_elements = src.num_elements();
    _data = aligned_alloc<T>(_num_elements);
    std::copy(src.begin(), src.end(), _data);
    _dims = src.dims();
    _strides = src.strides();
    _extents.init(_data, _dims, _strides);
  }

  tensor(tensor&& src) noexcept {
    _num_elements = src.num_elements();
    _data = src.data();
    src.data() = nullptr;
    _dims = std::move(src.dims());
    _strides = std::move(src.strides());
    _extents.init(_data, _dims, _strides);
  }

  template <typename L, typename Op, typename R, typename ITYPE>
  tensor(const Expr<L, Op, R, value_type, ITYPE>& rhs)
  : _data(nullptr)
  , _dims(D)
  , _strides(D) {
    static_assert(std::is_same<_internal_t, ITYPE>::value, "tensor assignment error: different internal type");
    _num_elements = rhs.num_elements();
    // _data = new T[_num_elements];
    _data = aligned_alloc<T>(_num_elements);
    _dims = rhs.dims();
    _strides[D - 1] = _num_elements / _dims[D - 1];
    for (int i = D - 2; i >= 0; i--) {
      _strides[i] = _strides[i + 1] / _dims[i];
    }
    _extents.init(_data, _dims, _strides);

    for (int i = 0; i < (int)_num_elements; i++)
      _data[i] = rhs.eval(i);
  }

  /*--- operators ---*/
  tensor& operator=(const tensor& src) {
    if (_num_elements != src.num_elements() && _data != nullptr) {
      aligned_deleter(_data);
      _num_elements = src.num_elements();
      _data = aligned_alloc<T>(_num_elements);
    }
    std::copy(src.begin(), src.end(), _data);
    _dims = src.dims();
    _strides = src.strides();
    _extents.init(_data, _dims, _strides);
    return *this;
  }

  tensor& operator=(tensor&& src) {
    // delete[] _data;
    aligned_deleter(_data);
    _num_elements = src.num_elements();
    _data = src.data();
    src.data() = nullptr;
    _dims = std::move(src.dims());
    _strides = std::move(src.strides());
    _extents.init(_data, _dims, _strides);
    return *this;
  }

  /*! EXPERIMNET */
  template <typename L, typename Op, typename R, typename ITYPE>
  tensor& operator=(const Expr<L, Op, R, value_type, ITYPE>& rhs) {
    static_assert(std::is_same<_internal_t, ITYPE>::value, "tensor assignment error: different internal type");
    if (rhs.num_elements() != num_elements()) {
      std::cerr << "tensor assignment error: num of elements miss-matched!" << std::endl;
      std::exit(1);
    }

    for (auto i = to_index(0); i < _num_elements; i++)
      _data[i] = rhs.eval(i);
    return *this;
  }

  template <typename L, typename Op, typename R, typename ITYPE>
  tensor& operator+=(const Expr<L, Op, R, value_type, ITYPE>& rhs) {
    static_assert(std::is_same<_internal_t, ITYPE>::value, "tensor assignment error: different internal type");
    if (rhs.num_elements() != num_elements()) {
      std::cerr << "tensor assignment error: num of elements miss-matched!" << std::endl;
      std::exit(1);
    }
    for (auto i = to_index(0); i < _num_elements; i++)
      _data[i] += rhs.eval(i);
    return *this;
  }

  template <typename L, typename Op, typename R, typename ITYPE>
  tensor& operator-=(const Expr<L, Op, R, value_type, ITYPE>& rhs) {
    static_assert(std::is_same<_internal_t, ITYPE>::value, "tensor assignment error: different internal type");
    if (rhs.num_elements() != num_elements()) {
      std::cerr << "tensor assignment error: num of elements miss-matched!" << std::endl;
      std::exit(1);
    }
    for (auto i = to_index(0); i < _num_elements; i++)
      _data[i] -= rhs.eval(i);
    return *this;
  }

  template <typename L, typename Op, typename R, typename ITYPE>
  tensor& operator/=(const Expr<L, Op, R, value_type, ITYPE>& rhs) {
    static_assert(std::is_same<_internal_t, ITYPE>::value, "tensor assignment error: different internal type");
    if (rhs.num_elements() != num_elements()) {
      std::cerr << "tensor assignment error: num of elements miss-matched!" << std::endl;
      std::exit(1);
    }
    for (auto i = to_index(0); i < _num_elements; i++)
      _data[i] /= rhs.eval(i);
    return *this;
  }

  template <typename L, typename Op, typename R, typename ITYPE>
  tensor& operator*=(const Expr<L, Op, R, value_type, ITYPE>& rhs) {
    static_assert(std::is_same<_internal_t, ITYPE>::value, "tensor assignment error: different internal type");
    if (rhs.num_elements() != num_elements()) {
      std::cerr << "tensor assignment error: num of elements miss-matched!" << std::endl;
      std::exit(1);
    }
    for (auto i = to_index(0); i < _num_elements; i++)
      _data[i] *= rhs.eval(i);
    return *this;
  }

  tensor_extent<T, D - 1, _internal_t>& operator[](const _internal_t i) {
    check_range(i, _dims[D - 1], D);
    return _extents.calc_index(i * _strides[D - 1]);
  }

  const tensor_extent<T, D - 1, _internal_t>& operator[](const _internal_t i) const {
    check_range(i, _dims[D - 1], D);
    return _extents.calc_index(i * _strides[D - 1]);
  }

  /*--- member functions ---*/
  T* begin() { return _data; }

  T* end() { return _data + _num_elements; }

  T* begin() const { return _data; }

  T* end() const { return _data + _num_elements; }

  T*& data() { return _data; }

  T* data() const { return _data; }

  _internal_t num_elements() const { return _num_elements; }

  tensor& reshape(const std::array<_internal_t, D>& shapes) {
    if (std::find(std::begin(shapes), std::end(shapes), 0) != std::end(shapes)) {
      std::cerr << "error: 0 is not permitted as a size of dimension" << std::endl;
      std::exit(1);
    }
    _num_elements = std::accumulate(std::begin(shapes), std::end(shapes), 1, std::multiplies<int>());
    std::copy(std::begin(shapes), std::end(shapes), std::begin(_dims));
    std::reverse(std::begin(_dims), std::end(_dims));
    _strides[D - 1] = _num_elements / _dims[D - 1];
    for (int i = D - 2; i >= 0; i--) {
      _strides[i] = _strides[i + 1] / _dims[i];
    }
    // delete[] _data;
    aligned_deleter(_data);
    //_data = new T[_num_elements];
    _data = aligned_alloc<T>(_num_elements);
    _extents.init(_data, _dims, _strides);

    return *this;
  }

  tensor& reshape(const _internal_t shape) {
    std::array<_internal_t, D> tmp;
    tmp.fill(1);
    tmp[D - 1] = shape;
    reshape(tmp);

    return *this;
  }

  template <typename... Args>
  tensor& reshape(Args... args) {
    reshape({static_cast<_internal_t>(args)...});
    return *this;
  }

  void as_shape_of(const std::array<_internal_t, D> shapes) {
    if (std::find(std::begin(shapes), std::end(shapes), 0) != std::end(shapes)) {
      std::cerr << "error: 0 is not permitted as a size of dimension" << std::endl;
      std::exit(1);
    }
    auto num_elements = std::accumulate(std::begin(shapes), std::end(shapes), 1, std::multiplies<int>());
    if ((int)_num_elements != num_elements) {
      std::cerr << "error: num. of elements miss-match" << std::endl;
      std::exit(1);
    }
    _num_elements = num_elements;
    std::copy(std::begin(shapes), std::end(shapes), std::begin(_dims));
    std::reverse(std::begin(_dims), std::end(_dims));
    _strides[D - 1] = _num_elements / _dims[D - 1];
    for (int i = D - 2; i >= 0; i--) {
      _strides[i] = _strides[i + 1] / _dims[i];
    }
    _extents.init(_data, _dims, _strides);
  }

  void as_shape_of(const _internal_t shape) {
    std::array<_internal_t, D> tmp;
    tmp.fill(1);
    tmp[D - 1] = shape;

    as_shape_of(tmp);
  }

  _internal_t shape() const { return D; }

  _internal_t shape(const _internal_t d) const { return _dims[d]; }

  template <std::size_t _D, typename std::enable_if<(_D <= D && _D >= 0), std::nullptr_t>::type = nullptr>
  _internal_t shape() {
    return _dims[_D];
  }

  const std::vector<_internal_t>& dims() const { return _dims; }

  _internal_t strides(const _internal_t d) const { return _strides[d]; }

  template <std::size_t _D, typename std::enable_if<(_D < D), std::nullptr_t>::type = nullptr>
  _internal_t strides() {
    return _strides[_D];
  }

  const std::vector<_internal_t>& strides() const { return _strides; }

  _internal_t dimension() const { return (_internal_t)D; }

  T& with_indices(const multi_index& indices) {
    for (int i = D - 1; i >= 0; --i)
      check_range(indices[D - 1 - i], _dims[i], i + 1);
    _internal_t idx = 0;
    for (int i = D - 1; i > 0; --i) {
      idx += indices[D - 1 - i] * strides(i);
    }
    idx += indices[D - 1];
    return _data[idx];
  }

  template <typename... Args>
  T& with_indices(Args... args) {
    static_assert(sizeof...(args) == D, "error: dimension miss-match");
    return with_indices({static_cast<_internal_t>(args)...});
  }

  template <typename... Args>
  T& operator()(Args... args) {
    static_assert(sizeof...(args) == D, "error: dimension miss-match");
    return with_indices({static_cast<_internal_t>(args)...});
  }

  T with_indices(const multi_index& indices) const {
    for (int i = D - 1; i >= 0; --i)
      check_range(indices[D - 1 - i], _dims[i], i + 1);
    _internal_t idx = 0;
    for (int i = D - 1; i > 0; --i) {
      idx += indices[D - 1 - i] * strides(i);
    }
    idx += indices[D - 1];
    return _data[idx];
  }

  template <typename... Args>
  T with_indices(Args... args) const {
    static_assert(sizeof...(args) == D, "error: dimension miss-match");
    return with_indices({static_cast<_internal_t>(args)...});
  }

  template <typename... Args>
  T operator()(Args... args) const {
    static_assert(sizeof...(args) == D, "error: dimension miss-match");
    return with_indices({static_cast<_internal_t>(args)...});
  }

  template <std::size_t _D>
  view<_D> make_view(const fixed_indices<D - _D>& indices) {
    static_assert(D - _D > 0, "dimension of view must be greater than 0.");
    multi_index midx;
    std::fill(midx.begin(), midx.end(), 0);
    std::copy(indices.begin(), indices.end(), midx.begin());
    return view<_D>(&with_indices(midx), _dims, _strides);
  }

  template <std::size_t _D, typename... Args>
  view<_D> make_view(Args... args) {
    return make_view<_D>({static_cast<_internal_t>(args)...});
  }

  template <typename type>
  _internal_t to_index(const type i) {
    return static_cast<_internal_t>(i);
  }

  template <typename type>
  tensor& fill(type x) {
    std::fill(_data, _data + _num_elements, (T)x);
    return *this;
  }

  ~tensor() {
    aligned_deleter(_data);
    // delete[] _data;
  }
};

template <typename T, typename INTERNAL_TYPE>
class tensor<T,
             1,
             INTERNAL_TYPE,
             typename std::enable_if<std::is_arithmetic<T>::value || std::is_pointer<T>::value, void>::type>
: public tensor_internal {
 public:
  typedef INTERNAL_TYPE _internal_t;
  typedef T value_type;

 private:
  T* _data;                          // a pointer to the data
  _internal_t _num_elements;         // total elements of tensor
  std::vector<_internal_t> _dims;    // the num of elements in each dimension
  std::vector<_internal_t> _strides; // access strides

  T eval(const int i) const { return _data[i]; }

 public:
  /*--- public typedefs ---*/
  typedef T type;
  typedef _internal_t multi_index;
  typedef _internal_t index_t;

  template <typename L, typename Op, typename R, typename TT, typename ITYPE>
  friend class Expr;
  // default constructor
  tensor()
  : _data(nullptr)
  , _dims(1)
  , _strides(1) {}

  // constructor: acceptes initililzer_list whose size is D
  tensor(_internal_t d)
  : _data(nullptr)
  , _dims(1)
  , _strides(1) {
    // error check
    // if one of arguments is 0
    if (d == 0) {
      std::cerr << "error: 0 is not permitted as a size of dimension" << std::endl;
      std::exit(1);
    }
    _num_elements = d;
    //_data = new T[_num_elements];
    _data = aligned_alloc<T>(_num_elements);
    _dims[0] = _num_elements;
    _strides[0] = 1;
  }

  tensor(const tensor& src) {
    _num_elements = src.num_elements();
    //_data = new T[_num_elements];
    _data = aligned_alloc<T>(_num_elements);
    std::copy(src.begin(), src.end(), _data);
    _dims = src.dims();
    _strides = src.strides();
  }

  tensor(const tensor_view<T, 1, _internal_t>& view)
  : _data(nullptr)
  , _dims(1)
  , _strides(1) {
    _num_elements = view.num_elements();
    _dims = view.dims();
    _strides = view.strides();
    //_data = new T[_num_elements];
    _data = aligned_alloc<T>(_num_elements);
    std::copy(std::begin(view), std::end(view), _data);
  }

  tensor(tensor&& src) noexcept {
    _num_elements = src.num_elements();
    _data = src.data();
    src.data() = nullptr;
    _dims = std::move(src.dims());
    _strides = std::move(src.strides());
  }

  template <typename L, typename Op, typename R, typename ITYPE>
  tensor(const Expr<L, Op, R, value_type, ITYPE>& rhs)
  : _data(nullptr)
  , _dims(1)
  , _strides(1) {
    static_assert(std::is_same<_internal_t, ITYPE>::value, "tensor assignment error: different internal type");
    _num_elements = rhs.num_elements();
    //_data = new T[_num_elements];
    _data = aligned_alloc<T>(_num_elements);
    _dims = rhs.dims();
    _strides[0] = 1;

    for (int i = 0; i < (int)_num_elements; i++)
      _data[i] = rhs.eval(i);
  }

  tensor& operator=(const tensor& src) {
    if (_num_elements != src.num_elements() && _data != nullptr) {
      aligned_deleter(_data);
      _num_elements = src.num_elements();
      _data = aligned_alloc<T>(_num_elements);
    }
    std::copy(src.begin(), src.end(), _data);
    _dims = src.dims();
    _strides = src.strides();
    return *this;
  }

  tensor& operator=(tensor&& src) {
    // delete[] _data;
    aligned_deleter(_data);
    _num_elements = src.num_elements();
    _data = src.data();
    src.data() = nullptr;
    _dims = std::move(src.dims());
    _strides = std::move(src.strides());

    return *this;
  }

  template <typename L, typename Op, typename R, typename ITYPE>
  tensor& operator=(const Expr<L, Op, R, value_type, ITYPE>& rhs) {
    static_assert(std::is_same<_internal_t, ITYPE>::value, "tensor assignment error: different internal type");
    if (rhs.num_elements() != num_elements()) {
      std::cerr << "tensor assignment error: num of elements miss-matched!" << std::endl;
      std::exit(1);
    }
    for (int i = 0; i < (int)_num_elements; i++)
      _data[i] = rhs.eval(i);
    return *this;
  }

  template <typename L, typename Op, typename R, typename ITYPE>
  tensor& operator+=(const Expr<L, Op, R, value_type, ITYPE>& rhs) {
    static_assert(std::is_same<_internal_t, ITYPE>::value, "tensor assignment error: different internal type");
    if (rhs.num_elements() != num_elements()) {
      std::cerr << "tensor assignment error: num of elements miss-matched!" << std::endl;
      std::exit(1);
    }
    for (int i = 0; i < (int)_num_elements; i++)
      _data[i] += rhs.eval(i);
    return *this;
  }

  template <typename L, typename Op, typename R, typename ITYPE>
  tensor& operator-=(const Expr<L, Op, R, value_type, ITYPE>& rhs) {
    static_assert(std::is_same<_internal_t, ITYPE>::value, "tensor assignment error: different internal type");
    if (rhs.num_elements() != num_elements()) {
      std::cerr << "tensor assignment error: num of elements miss-matched!" << std::endl;
      std::exit(1);
    }
    for (int i = 0; i < (int)_num_elements; i++)
      _data[i] -= rhs.eval(i);
    return *this;
  }

  template <typename L, typename Op, typename R, typename ITYPE>
  tensor& operator/=(const Expr<L, Op, R, value_type, ITYPE>& rhs) {
    static_assert(std::is_same<_internal_t, ITYPE>::value, "tensor assignment error: different internal type");
    if (rhs.num_elements() != num_elements()) {
      std::cerr << "tensor assignment error: num of elements miss-matched!" << std::endl;
      std::exit(1);
    }
    for (int i = 0; i < (int)_num_elements; i++)
      _data[i] /= rhs.eval(i);
    return *this;
  }

  template <typename L, typename Op, typename R, typename ITYPE>
  tensor& operator*=(const Expr<L, Op, R, value_type, ITYPE>& rhs) {
    static_assert(std::is_same<_internal_t, ITYPE>::value, "tensor assignment error: different internal type");
    if (rhs.num_elements() != num_elements()) {
      std::cerr << "tensor assignment error: num of elements miss-matched!" << std::endl;
      std::exit(1);
    }
    for (int i = 0; i < (int)_num_elements; i++)
      _data[i] *= rhs.eval(i);
    return *this;
  }

  T& operator[](const _internal_t i) {
    check_range(i, _dims[0], 1);
    return _data[i];
  }

  const T operator[](const _internal_t i) const {
    check_range(i, _dims[0], 1);
    return _data[i];
  }

  T* begin() { return _data; }

  T* end() { return _data + _num_elements; }

  T* begin() const { return _data; }

  T* end() const { return _data + _num_elements; }

  T*& data() { return _data; }

  T* data() const { return _data; }

  _internal_t num_elements() const { return _num_elements; }

  _internal_t shape() const { return 1; }

  _internal_t shape(const _internal_t d) const { return _dims[d]; }

  template <std::size_t _D, typename std::enable_if<(_D == 0), std::nullptr_t>::type = nullptr>
  _internal_t shape() {
    return _dims[_D];
  }

  tensor& reshape(const std::array<_internal_t, 1>& shapes) {
    if (std::find(std::begin(shapes), std::end(shapes), 0) != std::end(shapes)) {
      std::cerr << "error: 0 is not permitted as a size of dimension" << std::endl;
      std::exit(1);
    }
    _num_elements = shapes[0];
    std::copy(std::begin(shapes), std::end(shapes), std::begin(_dims));
    // std::reverse(std::begin(_dims), std::end(_dims));
    _strides[0] = 1; //_num_elements / _dims[D - 1];
    /*
    for (int i = D - 2; i >= 0; i--) {
      _strides[i] = _strides[i + 1] / _dims[i];
    }
    */
    // delete[] _data;
    aligned_deleter(_data);
    //_data = new T[_num_elements];
    _data = aligned_alloc<T>(_num_elements);
    //_extents.init(_data, _dims, _strides);

    return *this;
  }

  tensor& reshape(const _internal_t shape) {
    std::array<_internal_t, 1> tmp;
    tmp.fill(1);
    tmp[0] = shape;
    reshape(tmp);

    return *this;
  }

  template <typename... Args>
  tensor& reshape(Args... args) {
    reshape({static_cast<_internal_t>(args)...});
    return *this;
  }

  const std::vector<_internal_t>& dims() const { return _dims; }

  _internal_t strides(const _internal_t d) const { return _strides[d]; }

  const std::vector<_internal_t>& strides() const { return _strides; }

  _internal_t dimension() const { return (_internal_t)1; }

  T& with_indices(const _internal_t indices) {
    check_range(indices, _dims[0], 1);
    return _data[indices];
  }

  T with_indices(const _internal_t indices) const {
    check_range(indices, _dims[0], 1);
    return _data[indices];
  }

  T& operator()(const _internal_t indices) {
    check_range(indices, _dims[0], 1);
    return _data[indices];
  }

  T operator()(const _internal_t indices) const {
    check_range(indices, _dims[0], 1);
    return _data[indices];
  }

  template <typename type>
  _internal_t to_index(const type i) {
    return static_cast<_internal_t>(i);
  }

  template <typename type>
  tensor& fill(type x) {
    std::fill(_data, _data + _num_elements, (T)x);
    return *this;
  }

  ~tensor() { // delete[] _data;
    aligned_deleter(_data);
  }
};

/*--- typedefs ---*/
template <typename T, typename INTERNAL_TYPE = TENSOR_DEFAULT_INTERNAL_TYPE>
using vector = tensor<T, 1, INTERNAL_TYPE>;

} // namespace TENSOR_NAMESPACE_NAME

/*--- Expression Templates ---*/
namespace TENSOR_NAMESPACE_NAME {
template <typename L, typename Op, typename R, typename T, typename ITYPE>
class Expr : public tensor_internal {
 public:
  typedef ITYPE _internal_t;

 private:
  const L& _lhs;
  const R& _rhs;

  _internal_t _num_elements;
  const std::vector<_internal_t>& _dims;
  _internal_t _dimension;

 public:
  typedef T value_type;

  /*
    expr(const l& lhs, const r& rhs)
    : _lhs(lhs)
    , _rhs(rhs) {}
    */

  template <typename LL = L,
            typename RR = R,
            typename std::enable_if<(std::is_base_of<tensor_internal, LL>::value && std::is_arithmetic<RR>::value),
                                    std::nullptr_t>::type = nullptr>
  Expr(const L& lhs, const R& rhs)
  : _lhs(lhs)
  , _rhs(rhs)
  , _num_elements(lhs.num_elements())
  , _dims(lhs.dims())
  , _dimension(lhs.dimension()) {}

  template <typename LL = L,
            typename RR = R,
            typename std::enable_if<(std::is_base_of<tensor_internal, RR>::value && std::is_arithmetic<LL>::value),
                                    std::nullptr_t>::type = nullptr>
  Expr(const L& lhs, const R& rhs)
  : _lhs(lhs)
  , _rhs(rhs)
  , _num_elements(rhs.num_elements())
  , _dims(rhs.dims())
  , _dimension(rhs.dimension()) {}

  template <typename LL = L,
            typename RR = R,
            typename std::enable_if<(std::is_base_of<tensor_internal, LL>::value &&
                                     std::is_base_of<tensor_internal, RR>::value),
                                    std::nullptr_t>::type = nullptr>
  Expr(const L& lhs, const R& rhs)
  : _lhs(lhs)
  , _rhs(rhs)
  , _num_elements(lhs.num_elements())
  , _dims(lhs.dimension() > rhs.dimension() ? lhs.dims() : rhs.dims())
  , _dimension(lhs.dimension() > rhs.dimension() ? lhs.dimension() : rhs.dimension()) {
    if (_lhs.num_elements() != _rhs.num_elements()) {
      std::cerr << "tensor error: num of elements miss-matched! " << std::endl;
      std::exit(1);
    }
    _num_elements = _lhs.num_elements();
  }

  const L& lhs() const { return _lhs; }
  const R& rhs() const { return _rhs; }

  template <typename LL = L,
            typename RR = R,
            typename std::enable_if<std::is_base_of<tensor_internal, LL>::value, std::nullptr_t>::type = nullptr,
            typename std::enable_if<std::is_base_of<tensor_internal, RR>::value, std::nullptr_t>::type = nullptr>
  T eval(const int i) const {
    return Op::apply(_lhs.eval(i), _rhs.eval(i));
  }

  template <typename LL = L,
            typename RR = R,
            typename std::enable_if<std::is_arithmetic<LL>::value, std::nullptr_t>::type = nullptr,
            typename std::enable_if<std::is_base_of<tensor_internal, RR>::value, std::nullptr_t>::type = nullptr>
  T eval(const int i) const {
    return Op::apply((T)_lhs, _rhs.eval(i));
  }

  template <typename LL = L,
            typename RR = R,
            typename std::enable_if<std::is_arithmetic<RR>::value, std::nullptr_t>::type = nullptr,
            typename std::enable_if<std::is_base_of<tensor_internal, LL>::value, std::nullptr_t>::type = nullptr>
  T eval(const int i) const {
    return Op::apply(_lhs.eval(i), (T)_rhs);
  }

  /*--- functions ---*/
  _internal_t num_elements() const { return _num_elements; }
  const std::vector<_internal_t>& dims() const { return _dims; }
  _internal_t dimension() const { return _dimension; }
};

struct Plus {
  template <typename T>
  static inline T apply(const T lhs, const T rhs) {
    return lhs + rhs;
  }
};

struct Minus {
  template <typename T>
  static inline T apply(const T lhs, const T rhs) {
    return lhs - rhs;
  }
};

struct Div {
  template <typename T>
  static inline T apply(const T lhs, const T rhs) {
    return lhs / rhs;
  }
};

struct Mul {
  template <typename T>
  static inline T apply(const T lhs, const T rhs) {
    return lhs * rhs;
  }
};

/*--- Operators for Expression templates ---*/

/*
  3パターンで場合分け
  1. 両辺がtensor_internalを継承している場合
  2. 左辺がtensor_internalを継承している場合
  3. 右辺がtensor_internalを継承している場合
*/

/* Plus */
template <typename L,
          typename R,
          typename std::enable_if<std::is_base_of<tensor_internal, L>::value, std::nullptr_t>::type = nullptr,
          typename std::enable_if<std::is_base_of<tensor_internal, R>::value, std::nullptr_t>::type = nullptr,
          typename std::enable_if<std::is_same<typename L::_internal_t, typename R::_internal_t>::value,
                                  std::nullptr_t>::type = nullptr>
Expr<L, Plus, R, typename L::value_type> operator+(const L& lhs, const R& rhs) {
  return Expr<L, Plus, R, typename L::value_type, typename L::_internal_t>(lhs, rhs);
}

template <typename L,
          typename R,
          typename std::enable_if<std::is_arithmetic<L>::value, std::nullptr_t>::type = nullptr,
          typename std::enable_if<std::is_base_of<tensor_internal, R>::value, std::nullptr_t>::type = nullptr>
Expr<L, Plus, R, typename R::value_type> operator+(const L& lhs, const R& rhs) {
  return Expr<L, Plus, R, typename R::value_type, typename R::_internal_t>(lhs, rhs);
}

template <typename L,
          typename R,
          typename std::enable_if<std::is_arithmetic<R>::value, std::nullptr_t>::type = nullptr,
          typename std::enable_if<std::is_base_of<tensor_internal, L>::value, std::nullptr_t>::type = nullptr>
Expr<L, Plus, R, typename L::value_type> operator+(const L& lhs, const R& rhs) {
  return Expr<L, Plus, R, typename L::value_type, typename L::_internal_t>(lhs, rhs);
}

/* Minus */
template <typename L,
          typename R,
          typename std::enable_if<std::is_base_of<tensor_internal, L>::value, std::nullptr_t>::type = nullptr,
          typename std::enable_if<std::is_base_of<tensor_internal, R>::value, std::nullptr_t>::type = nullptr,
          typename std::enable_if<std::is_same<typename L::_internal_t, typename R::_internal_t>::value,
                                  std::nullptr_t>::type = nullptr>
Expr<L, Minus, R, typename L::value_type> operator-(const L& lhs, const R& rhs) {
  return Expr<L, Minus, R, typename L::value_type, typename L::_internal_t>(lhs, rhs);
}

template <typename L,
          typename R,
          typename std::enable_if<std::is_arithmetic<L>::value, std::nullptr_t>::type = nullptr,
          typename std::enable_if<std::is_base_of<tensor_internal, R>::value, std::nullptr_t>::type = nullptr>
Expr<L, Minus, R, typename R::value_type> operator-(const L& lhs, const R& rhs) {
  return Expr<L, Minus, R, typename R::value_type, typename R::_internal_t>(lhs, rhs);
}

template <typename L,
          typename R,
          typename std::enable_if<std::is_arithmetic<R>::value, std::nullptr_t>::type = nullptr,
          typename std::enable_if<std::is_base_of<tensor_internal, L>::value, std::nullptr_t>::type = nullptr>
Expr<L, Minus, R, typename L::value_type> operator-(const L& lhs, const R& rhs) {
  return Expr<L, Minus, R, typename L::value_type, typename L::internal_t>(lhs, rhs);
}

/* Div*/
template <typename L,
          typename R,
          typename std::enable_if<std::is_base_of<tensor_internal, L>::value, std::nullptr_t>::type = nullptr,
          typename std::enable_if<std::is_base_of<tensor_internal, R>::value, std::nullptr_t>::type = nullptr,
          typename std::enable_if<std::is_same<typename L::_internal_t, typename R::_internal_t>::value,
                                  std::nullptr_t>::type = nullptr>
Expr<L, Div, R, typename L::value_type> operator/(const L& lhs, const R& rhs) {
  return Expr<L, Div, R, typename L::value_type, typename L::_internal_t>(lhs, rhs);
}

template <typename L,
          typename R,
          typename std::enable_if<std::is_arithmetic<L>::value, std::nullptr_t>::type = nullptr,
          typename std::enable_if<std::is_base_of<tensor_internal, R>::value, std::nullptr_t>::type = nullptr>
Expr<L, Div, R, typename R::value_type> operator/(const L& lhs, const R& rhs) {
  return Expr<L, Div, R, typename R::value_type, typename R::_internal_t>(lhs, rhs);
}

template <typename L,
          typename R,
          typename std::enable_if<std::is_arithmetic<R>::value, std::nullptr_t>::type = nullptr,
          typename std::enable_if<std::is_base_of<tensor_internal, L>::value, std::nullptr_t>::type = nullptr>
Expr<L, Div, R, typename L::value_type> operator/(const L& lhs, const R& rhs) {
  return Expr<L, Div, R, typename L::value_type, typename L::_internal_t>(lhs, rhs);
}

/* Mul */
template <typename L,
          typename R,
          typename std::enable_if<std::is_base_of<tensor_internal, L>::value, std::nullptr_t>::type = nullptr,
          typename std::enable_if<std::is_base_of<tensor_internal, R>::value, std::nullptr_t>::type = nullptr,
          typename std::enable_if<std::is_same<typename L::_internal_t, typename R::_internal_t>::value,
                                  std::nullptr_t>::type = nullptr>
Expr<L, Mul, R, typename L::value_type> operator*(const L& lhs, const R& rhs) {
  return Expr<L, Mul, R, typename L::value_type, typename L::_internal_t>(lhs, rhs);
}

template <typename L,
          typename R,
          typename std::enable_if<std::is_arithmetic<L>::value, std::nullptr_t>::type = nullptr,
          typename std::enable_if<std::is_base_of<tensor_internal, R>::value, std::nullptr_t>::type = nullptr>
Expr<L, Mul, R, typename R::value_type> operator*(const L& lhs, const R& rhs) {
  return Expr<L, Mul, R, typename R::value_type, typename R::_internal_t>(lhs, rhs);
}

template <typename L,
          typename R,
          typename std::enable_if<std::is_arithmetic<R>::value, std::nullptr_t>::type = nullptr,
          typename std::enable_if<std::is_base_of<tensor_internal, L>::value, std::nullptr_t>::type = nullptr>
Expr<L, Mul, R, typename L::value_type> operator*(const L& lhs, const R& rhs) {
  return Expr<L, Mul, R, typename L::value_type, typename L::_internal_t>(lhs, rhs);
}

} // namespace TENSOR_NAMESPACE_NAME