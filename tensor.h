/*  MIT License

    Copyright (c) [2019] [Yuta Kambara]

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

/*  Configurable MACROs:
      TENSOR_ENABLE_ASSERTS
        - enable range check when access to elements.
          (cause performance overhead)
      TENSOR_DEFAULT_INTERNAL_TYPE = T
        - use T as internal index type.
          T must be integer type.
          (default is std::size_t)
*/

#pragma once

/*--- check if the compiler supports C++11 ---*/
#if __cplusplus < 201103L
#error Tensor library needs at least a C++11 compliant compiler
#endif

/*--- if TENSOR_ENABLE_ASSERTS macro is NOT defined ---*/
#if not defined(TENSOR_ENABLE_ASSERTS)
#define NDEBUG
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
#define TENSOR_DEFAULT_INTERNAL_TYPE std::size_t
#endif

#include <algorithm>
#include <array>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <vector>

namespace rnz {

/* the bese class of all tensor classes */
class tensor_internal {};

/*--- forward declarations ---*/
template <typename T, std::size_t D, typename INTERNAL_TYPE = TENSOR_DEFAULT_INTERNAL_TYPE>
struct tensor_extent;
template <typename T,
          std::size_t D,
          typename INTERNAL_TYPE = TENSOR_DEFAULT_INTERNAL_TYPE,
          typename = void>
class tensor;
template <typename T, std::size_t D, typename INTERNAL_TYPE = TENSOR_DEFAULT_INTERNAL_TYPE>
class tensor_view;

template <typename L, typename Op, typename R, typename T>
class Expr;

/*--- range check functions ---*/
static inline constexpr bool is_asserts_enabled() {
#ifdef TENSOR_ENABLE_ASSERTS
  return true;
#else
  return false;
#endif
}

template <typename T1,
          typename T2,
          typename std::enable_if<std::is_integral<T1>::value && std::is_integral<T2>::value,
                                  std::nullptr_t>::type = nullptr>
void check_range(const T1 i, const T2 dim) {
  if (is_asserts_enabled())
    if (unlikely(i < 0 || i >= dim)) {
      std::cerr << "error: out of range access (idx=" << i << ", dim=" << dim << ")" << std::endl;
      std::exit(1);
    }
}

template <typename T1,
          typename T2,
          typename T3,
          typename std::enable_if<std::is_integral<T1>::value && std::is_integral<T2>::value &&
                                      std::is_integral<T3>::value,
                                  std::nullptr_t>::type = nullptr>
void check_range(const T1 i, const T2 dim, const T3 D) {
  if (is_asserts_enabled())
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

/*--- functions ---*/

// a helper function returning a tensor object.
template <typename T, std::size_t D, typename INTERNAL_TYPE = TENSOR_DEFAULT_INTERNAL_TYPE>
tensor<T, D, INTERNAL_TYPE> make_tensor(const std::initializer_list<INTERNAL_TYPE>& initialzier) {
  return tensor<T, D, INTERNAL_TYPE>(initialzier);
}

/*--- class and struct ---*/
template <typename T, std::size_t D, typename INTERNAL_TYPE>
class tensor_view : public tensor_internal {
 private:
  typedef INTERNAL_TYPE _internal_t;
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

  template <typename L, typename Op, typename R, typename TT>
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

  const T* const begin() const { return _data; }

  const T* const end() const { return _data + _num_elements; }

  T*& data() { return _data; }

  T* const& data() const { return _data; }

  _internal_t num_elements() const { return _num_elements; }

  _internal_t shape() const { return D; }

  _internal_t shape(const _internal_t d) const { return _dims[d - 1]; }

  const std::vector<_internal_t>& dims() const { return _dims; }

  _internal_t strides(const _internal_t d) const { return _strides[d]; }

  const std::vector<_internal_t>& strides() const { return _strides; }

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
};

template <typename T, typename INTERNAL_TYPE>
class tensor_view<T, 1, INTERNAL_TYPE> : public tensor_internal {
 private:
  typedef INTERNAL_TYPE _internal_t;
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

  template <typename L, typename Op, typename R, typename TT>
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

  const T* const begin() const { return _data; }

  const T* const end() const { return _data + _num_elements; }

  T*& data() { return _data; }

  T* const data() const { return _data; }

  void fill(T x) { std::fill(_data, _data + _num_elements, x); }

  _internal_t num_elements() const { return _num_elements; }

  _internal_t shape() const { return 1; }

  _internal_t shape(const _internal_t d) const { return _dims[d - 1]; }

  const std::vector<_internal_t>& dims() const { return _dims; }

  _internal_t strides(const _internal_t d) const { return _strides[d]; }

  const std::vector<_internal_t>& strides() const { return _strides; }

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
};

template <typename T, std::size_t D, typename INTERNAL_TYPE>
struct tensor_extent : public tensor_internal {
  typedef INTERNAL_TYPE _internal_t;
  tensor_extent<T, D - 1, _internal_t> _extents;
  T* _p;
  _internal_t _dim;    // D次元の要素数
  _internal_t _stride; // D-1次元までの要素数
  mutable _internal_t _accum;

  //  T eval(const int i) const { return _data[i]; }

  template <typename L, typename Op, typename R, typename TT>
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
  : _p(p) {}

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
class tensor<
    T,
    D,
    INTERNAL_TYPE,
    typename std::enable_if<std::is_arithmetic<T>::value || std::is_pointer<T>::value, void>::type>
: public tensor_internal {
 private:
  typedef INTERNAL_TYPE _internal_t;
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

  template <typename L, typename Op, typename R, typename TT>
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
    _data = new T[_num_elements];
    _strides[D - 1] = _num_elements / _dims[D - 1];
    for (int i = D - 2; i >= 0; i--) {
      _strides[i] = _strides[i + 1] / _dims[i];
    }
    _extents.init(_data, _dims, _strides);
  }

  template <typename... Args>
  tensor(Args... args)
  : tensor({static_cast<_internal_t>(args)...}){};

  tensor(const tensor_view<T, D, _internal_t>& view)
  : _data(nullptr)
  , _dims(D)
  , _strides(D) {
    _num_elements = view.num_elements();
    _dims = view.dims();
    _strides = view.strides();
    _data = new T[_num_elements];
    std::copy(view.begin(), view.end(), _data);
    _extents.init(_data, _dims, _strides);
  }

  tensor(const tensor& src) {
    _num_elements = src.num_elements();
    _data = new T[_num_elements];
    std::copy(src.begin(), src.end(), _data);
    _dims = src.dims();
    _strides = src.strides();
    _extents.init(_data, _dims, _strides);
  }

  tensor(tensor&& src) {
    _num_elements = src.num_elements();
    _data = src.data();
    src.data() = nullptr;
    _dims = std::move(src.dims());
    _strides = std::move(src.strides());
    _extents.init(_data, _dims, _strides);
  }

  /*--- operators ---*/
  tensor& operator=(const tensor& src) {
    delete[] _data;
    _num_elements = src.num_elements();
    _data = new T[_num_elements];
    std::copy(src.begin(), src.end(), _data);
    _dims = src.dims();
    _strides = src.strides();
    _extents.init(_data, _dims, _strides);
    return *this;
  }

  tensor& operator=(tensor&& src) {
    delete[] _data;
    _num_elements = src.num_elements();
    _data = src.data();
    _dims = std::move(src.dims());
    _strides = std::move(src.strides());
    _extents.init(_data, _dims, _strides);
    return *this;
  }

  /*! EXPERIMNET */
  template <typename L, typename Op, typename R>
  tensor& operator=(const Expr<L, Op, R, value_type>& rhs) {
    for (int i = 0; i < _num_elements; i++)
      _data[i] = rhs.eval(i);
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

  const T* const begin() const { return _data; }

  const T* const end() const { return _data + _num_elements; }

  T*& data() { return _data; }

  const T* const data() const { return _data; }

  void fill(T x) { std::fill(_data, _data + _num_elements, x); }

  _internal_t num_elements() const { return _num_elements; }

  void reshape(const std::array<_internal_t, D>& shapes) {
    if (std::find(std::begin(shapes), std::end(shapes), 0) != std::end(shapes)) {
      std::cerr << "error: 0 is not permitted as a size of dimension" << std::endl;
      std::exit(1);
    }
    _num_elements =
        std::accumulate(std::begin(shapes), std::end(shapes), 1, std::multiplies<int>());
    std::copy(std::begin(shapes), std::end(shapes), std::begin(_dims));
    std::reverse(std::begin(_dims), std::end(_dims));
    _strides[D - 1] = _num_elements / _dims[D - 1];
    for (int i = D - 2; i >= 0; i--) {
      _strides[i] = _strides[i + 1] / _dims[i];
    }
    delete[] _data;
    _data = new T[_num_elements];
    _extents.init(_data, _dims, _strides);
  }

  void reshape(const _internal_t shape) {
    std::array<_internal_t, D> tmp;
    tmp.fill(1);
    tmp[D - 1] = shape;
    reshape(tmp);
  }

  template <typename... Args>
  void reshape(Args... args) {
    reshape({static_cast<_internal_t>(args)...});
  }

  void as_shape_of(const std::array<_internal_t, D> shapes) {
    if (std::find(std::begin(shapes), std::end(shapes), 0) != std::end(shapes)) {
      std::cerr << "error: 0 is not permitted as a size of dimension" << std::endl;
      std::exit(1);
    }
    auto num_elements =
        std::accumulate(std::begin(shapes), std::end(shapes), 1, std::multiplies<int>());
    if (_num_elements != num_elements) {
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

  _internal_t shape(const _internal_t d) const { return _dims[d - 1]; }

  const std::vector<_internal_t>& dims() const { return _dims; }

  _internal_t strides(const _internal_t d) const { return _strides[d]; }

  const std::vector<_internal_t>& strides() const { return _strides; }

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

  ~tensor() { delete[] _data; }
};

template <typename T, typename INTERNAL_TYPE>
class tensor<
    T,
    1,
    INTERNAL_TYPE,
    typename std::enable_if<std::is_arithmetic<T>::value || std::is_pointer<T>::value, void>::type>
: public tensor_internal {
 private:
  typedef INTERNAL_TYPE _internal_t;
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

  template <typename L, typename Op, typename R, typename TT>
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
    _data = new T[_num_elements];
    _dims[0] = _num_elements;
    _strides[0] = 1;
  }

  tensor(const tensor& src) {
    _num_elements = src.num_elements();
    _data = new T[_num_elements];
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
    _data = new T[_num_elements];
    std::copy(std::begin(view), std::end(view), _data);
  }

  tensor(tensor&& src) {
    _num_elements = src.num_elements();
    _data = src.data();
    src.data() = nullptr;
    _dims = std::move(src.dims());
    _strides = std::move(src.strides());
  }

  tensor& operator=(const tensor& src) {
    delete[] _data;
    _num_elements = src.num_elements();
    _data = new T[_num_elements];
    std::copy(src.begin(), src.end(), _data);
    _dims = src.dims();
    _strides = src.strides();
    return *this;
  }

  tensor& operator=(tensor&& src) {
    delete[] _data;
    _num_elements = src.num_elements();
    _data = src.data();
    src.data() = nullptr;
    _dims = std::move(src.dims());
    _strides = std::move(src.strides());

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

  const T* const begin() const { return _data; }

  const T* const end() const { return _data + _num_elements; }

  T*& data() { return _data; }

  const T* const data() const { return _data; }

  void fill(T x) { std::fill(_data, _data + _num_elements, x); }

  _internal_t num_elements() const { return _num_elements; }

  _internal_t shape() const { return 1; }

  _internal_t shape(const _internal_t d) const { return _dims[d - 1]; }

  const std::vector<_internal_t>& dims() const { return _dims; }

  _internal_t strides(const _internal_t d) const { return _strides[d]; }

  const std::vector<_internal_t>& strides() const { return _strides; }

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

  ~tensor() { delete[] _data; }
};

/*--- typedefs ---*/
template <typename T, typename INTERNAL_TYPE = TENSOR_DEFAULT_INTERNAL_TYPE>
using vector = tensor<T, 1, INTERNAL_TYPE>;

} // namespace rnz

/*--- Expression Templates ---*/
namespace rnz {
template <typename L, typename Op, typename R, typename T>
class Expr : public tensor_internal {
  const L& _lhs;
  const R& _rhs;

 public:
  typedef T value_type;

  Expr(const L& lhs, const R& rhs)
  : _lhs(lhs)
  , _rhs(rhs) {}

  const L& lhs() { return _lhs; }
  const R& rhs() { return _rhs; }

  template <typename LL = L,
            typename RR = R,
            typename std::enable_if<std::is_convertible<LL, tensor_internal>::value,
                                    std::nullptr_t>::type = nullptr,
            typename std::enable_if<std::is_convertible<RR, tensor_internal>::value,
                                    std::nullptr_t>::type = nullptr>
  T eval(const int i) const {
    return Op::apply(_lhs.eval(i), _rhs.eval(i));
  }

  template <typename LL = L,
            typename RR = R,
            typename std::enable_if<std::is_arithmetic<LL>::value, std::nullptr_t>::type = nullptr,
            typename std::enable_if<std::is_convertible<RR, tensor_internal>::value,
                                    std::nullptr_t>::type = nullptr>
  T eval(const int i) const {
    return Op::apply((T)_lhs, _rhs.eval(i));
  }

  template <typename LL = L,
            typename RR = R,
            typename std::enable_if<std::is_arithmetic<RR>::value, std::nullptr_t>::type = nullptr,
            typename std::enable_if<std::is_convertible<LL, tensor_internal>::value,
                                    std::nullptr_t>::type = nullptr>
  T eval(const int i) const {
    return Op::apply(_lhs.eval(i), (T)_rhs);
  }
  //  T data(const int i) const { return _lhs.data()[i] + _rhs.data()[i]; }
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

/* Plus */
template <typename L,
          typename R,
          typename std::enable_if<std::is_convertible<L, tensor_internal>::value,
                                  std::nullptr_t>::type = nullptr,
          typename std::enable_if<std::is_convertible<R, tensor_internal>::value,
                                  std::nullptr_t>::type = nullptr>
Expr<L, Plus, R, typename L::value_type> operator+(const L& lhs, const R& rhs) {
  return Expr<L, Plus, R, typename L::value_type>(lhs, rhs);
}

template <typename L,
          typename R,
          typename std::enable_if<std::is_arithmetic<L>::value, std::nullptr_t>::type = nullptr,
          typename std::enable_if<std::is_convertible<R, tensor_internal>::value,
                                  std::nullptr_t>::type = nullptr>
Expr<L, Plus, R, typename R::value_type> operator+(const L& lhs, const R& rhs) {
  return Expr<L, Plus, R, typename R::value_type>(lhs, rhs);
}

template <typename L,
          typename R,
          typename std::enable_if<std::is_arithmetic<R>::value, std::nullptr_t>::type = nullptr,
          typename std::enable_if<std::is_convertible<L, tensor_internal>::value,
                                  std::nullptr_t>::type = nullptr>
Expr<L, Plus, R, typename L::value_type> operator+(const L& lhs, const R& rhs) {
  return Expr<L, Plus, R, typename L::value_type>(lhs, rhs);
}

/* Minus */
template <typename L,
          typename R,
          typename std::enable_if<std::is_convertible<L, tensor_internal>::value,
                                  std::nullptr_t>::type = nullptr,
          typename std::enable_if<std::is_convertible<R, tensor_internal>::value,
                                  std::nullptr_t>::type = nullptr>
Expr<L, Minus, R, typename L::value_type> operator-(const L& lhs, const R& rhs) {
  return Expr<L, Minus, R, typename L::value_type>(lhs, rhs);
}

template <typename L,
          typename R,
          typename std::enable_if<std::is_arithmetic<L>::value, std::nullptr_t>::type = nullptr,
          typename std::enable_if<std::is_convertible<R, tensor_internal>::value,
                                  std::nullptr_t>::type = nullptr>
Expr<L, Minus, R, typename R::value_type> operator-(const L& lhs, const R& rhs) {
  return Expr<L, Minus, R, typename R::value_type>(lhs, rhs);
}

template <typename L,
          typename R,
          typename std::enable_if<std::is_arithmetic<R>::value, std::nullptr_t>::type = nullptr,
          typename std::enable_if<std::is_convertible<L, tensor_internal>::value,
                                  std::nullptr_t>::type = nullptr>
Expr<L, Minus, R, typename L::value_type> operator-(const L& lhs, const R& rhs) {
  return Expr<L, Minus, R, typename L::value_type>(lhs, rhs);
}

/* Div*/
template <typename L,
          typename R,
          typename std::enable_if<std::is_convertible<L, tensor_internal>::value,
                                  std::nullptr_t>::type = nullptr,
          typename std::enable_if<std::is_convertible<R, tensor_internal>::value,
                                  std::nullptr_t>::type = nullptr>
Expr<L, Div, R, typename L::value_type> operator/(const L& lhs, const R& rhs) {
  return Expr<L, Div, R, typename L::value_type>(lhs, rhs);
}

template <typename L,
          typename R,
          typename std::enable_if<std::is_arithmetic<L>::value, std::nullptr_t>::type = nullptr,
          typename std::enable_if<std::is_convertible<R, tensor_internal>::value,
                                  std::nullptr_t>::type = nullptr>
Expr<L, Div, R, typename R::value_type> operator/(const L& lhs, const R& rhs) {
  return Expr<L, Div, R, typename R::value_type>(lhs, rhs);
}

template <typename L,
          typename R,
          typename std::enable_if<std::is_arithmetic<R>::value, std::nullptr_t>::type = nullptr,
          typename std::enable_if<std::is_convertible<L, tensor_internal>::value,
                                  std::nullptr_t>::type = nullptr>
Expr<L, Div, R, typename L::value_type> operator/(const L& lhs, const R& rhs) {
  return Expr<L, Div, R, typename L::value_type>(lhs, rhs);
}

/* Mul */
template <typename L,
          typename R,
          typename std::enable_if<std::is_convertible<L, tensor_internal>::value,
                                  std::nullptr_t>::type = nullptr,
          typename std::enable_if<std::is_convertible<R, tensor_internal>::value,
                                  std::nullptr_t>::type = nullptr>
Expr<L, Mul, R, typename L::value_type> operator*(const L& lhs, const R& rhs) {
  return Expr<L, Mul, R, typename L::value_type>(lhs, rhs);
}

template <typename L,
          typename R,
          typename std::enable_if<std::is_arithmetic<L>::value, std::nullptr_t>::type = nullptr,
          typename std::enable_if<std::is_convertible<R, tensor_internal>::value,
                                  std::nullptr_t>::type = nullptr>
Expr<L, Mul, R, typename R::value_type> operator*(const L& lhs, const R& rhs) {
  return Expr<L, Mul, R, typename R::value_type>(lhs, rhs);
}

template <typename L,
          typename R,
          typename std::enable_if<std::is_arithmetic<R>::value, std::nullptr_t>::type = nullptr,
          typename std::enable_if<std::is_convertible<L, tensor_internal>::value,
                                  std::nullptr_t>::type = nullptr>
Expr<L, Mul, R, typename L::value_type> operator*(const L& lhs, const R& rhs) {
  return Expr<L, Mul, R, typename L::value_type>(lhs, rhs);
}

} // namespace rnz