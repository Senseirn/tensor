#pragma once

/* By default, range check function is disabled.
   to enable the function,
   you should define the following macro
*/

/*  Configuration MACRO:
      TENSOR_ENABLE_ASSERTS
        disable range check when access to elements.
        (cause performace overhead)
*/

/*--- if TENSOR_ENABLE_ASSERTS macro is defined ---*/
#if not defined(TENSOR_ENABLE_ASSERTS)
#define NDEBUG
#endif

/*--- define likely and unlikely macros ---*/
#if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x) (x)
#define unlikely(x) (x)
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <vector>

namespace rnz {

/*--- check_range() ---*/
void check_range(const int i, const int dim) {
#ifdef TENSOR_ENABLE_ASSERTS
  if (unlikely(i < 0 || i >= dim)) {
    std::cerr << "error: out of range access (idx=" << i << ", dim=" << dim << ")" << std::endl;
    std::exit(1);
  }
#endif
}

void check_range(const int i, const int dim, const int D) {
#ifdef TENSOR_ENABLE_ASSERTS
  if (unlikely(i < 0 || i >= dim)) {
    std::cerr << "error: out of range access [trying to access " << i << "th element in " << D
              << "th dimension(max range = " << dim - 1 << ")]" << std::endl;
    std::exit(1);
  }
#endif
}

/*--- forward declarations ---*/
template <typename T, std::size_t D>
struct tensor_extent;
template <typename T, std::size_t D>
class tensor;
template <typename T, std::size_t D>
class tensor_view;

/*--- functions ---*/

// a helper function returning a tensor object.
template <typename T, std::size_t D>
tensor<T, D> make_tensor(const std::initializer_list<int>& initialzier) {
  return tensor<T, D>(initialzier);
}

/*--- class and struct ---*/

template <typename T, std::size_t D>
class tensor_view {
 private:
  T* _data; // a pointer to the first data of dimension D
  int _num_elements;
  std::vector<int> _dims;    // the num of elements in each dimension
  std::vector<int> _strides; // access strides
  tensor_extent<T, D - 1> _extents;

 public:
  /*--- typedefs ---*/
  typedef std::array<int, D> multi_index;
  template <std::size_t _D>
  using view = tensor_view<T, _D>;
  template <std::size_t _D>
  using fixed_indices = std::array<int, _D>;

  /*--- constructors ---*/
  tensor_view()
  : _data(nullptr)
  , _dims(D)
  , _strides(D) {}
  tensor_view(T* p, const std::vector<int>& dims, const std::vector<int>& strides)
  : _data(p)
  , _dims(D)
  , _strides(D) {
    const int size_diff = dims.size() - D;
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

  tensor_extent<T, D - 1>& operator[](int i) {
    check_range(i, _dims[D - 1], D);
    return _extents.calc_index(i * _strides[D - 1]);
  }

  const tensor_extent<T, D - 1>& operator[](int i) const {
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

  int num_elements() const { return _num_elements; }

  int shape() const { return D; }

  int shape(const int d) const { return _dims[d - 1]; }

  const std::vector<int>& dims() const { return _dims; }

  int strides(const int d) const { return _strides[d]; }

  const std::vector<int>& strides() const { return _strides; }

  T& with_indices(const multi_index& indices) {
    for (int i = D - 1; i >= 0; --i) {
      check_range(indices[D - 1 - i], _dims[i], i + 1);
    }
    int idx = 0;
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
    int idx = 0;
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

  tensor<T, D> to_tensor() const { return tensor<T, D>(*this); }

  template <std::size_t _D>
  view<_D> make_view(const fixed_indices<D - _D>& indices) {
    static_assert(D - _D > 0, "dimension of view must be greater than 0.");
    multi_index midx;
    std::fill(midx.begin(), midx.end(), 0);
    std::copy(indices.begin(), indices.end(), midx.begin());
    return view<_D>(&with_indices(midx), _dims, _strides);
  }
};

template <typename T>
class tensor_view<T, 1> {
 private:
  T* _data;                  // a pointer to the data
  std::size_t _num_elements; // total elements of tensor
  std::vector<int> _dims;    // the num of elements in each dimension
  std::vector<int> _strides; // access strides
  // tensor_extent<T, D - 1> _extents;  // inner struct to calculate index

 public:
  /*--- typedefs ---*/
  typedef T type;
  typedef int multi_index;

  /*--- constructors ---*/
  tensor_view()
  : _data(nullptr)
  , _num_elements(0)
  , _dims(1)
  , _strides(1) {}

  // constructor: acceptes initililzer_list whose size is D
  tensor_view(T* p, const std::vector<int>& dims, const std::vector<int>& strides)
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

  T& operator[](int i) {
    check_range(i, _dims[0], 1);
    return _data[i];
  }

  const T operator[](int i) const {
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

  int num_elements() const { return _num_elements; }

  int shape() const { return 1; }

  int shape(const int d) const { return _dims[d - 1]; }

  const std::vector<int>& dims() const { return _dims; }

  int strides(const int d) const { return _strides[d]; }

  const std::vector<int>& strides() const { return _strides; }

  T& with_indices(const int indices) {
    check_range(indices, _dims[0], 1);
    return _data[indices];
  }

  T with_indices(const int indices) const {
    check_range(indices, _dims[0], 1);
    return _data[indices];
  }

  tensor<T, 1> to_tensor() const { return tensor<T, 1>(*this); }
};

template <typename T, std::size_t D>
struct tensor_extent {
  tensor_extent<T, D - 1> _extents;
  T* _p;
  int _dim;    // D次元の要素数
  int _stride; // D-1次元までの要素数
  mutable int _accum;

  /*--- functions ---*/
  inline tensor_extent<T, D>& calc_index(int accum) {
    _accum = accum;
    return *this;
  }

  inline const tensor_extent<T, D>& calc_index(int accum) const {
    _accum = accum;
    return *this;
  }

  tensor_extent()
  : _p(nullptr)
  , _stride(0)
  , _extents(nullptr, 0) {}

  tensor_extent(T* p, int stride)
  : _p(p)
  , _stride(stride)
  , _extents(p, stride) {}

  tensor_extent<T, D - 1>& operator[](int i) {
    check_range(i, _dim, D);
    return _extents.calc_index(i * _stride + _accum);
  }

  const tensor_extent<T, D - 1>& operator[](int i) const {
    check_range(i, _dim, D);
    return _extents.calc_index(i * _stride + _accum);
  }

  int accum() const { return _accum; }

  void init(T* p, std::vector<int>& dims, std::vector<int>& strides) {
    _p = p;
    _dim = dims[D - 1];
    _stride = strides[D - 1];
    _extents.init(p, dims, strides);
  }
};

template <typename T>
struct tensor_extent<T, 1> {
  T* _p;
  int _dim;
  mutable int _accum;

  /*--- functions ---*/
  tensor_extent()
  : _p(nullptr) {}
  tensor_extent(T* p, int stride)
  : _p(p) {}

  tensor_extent<T, 1>& calc_index(int accum) {
    _accum = accum;
    return *this;
  }

  const tensor_extent<T, 1>& calc_index(int accum) const {
    _accum = accum;
    return *this;
  }

  void init(T* p, std::vector<int>& dims, std::vector<int>& strides) {
    _dim = dims[0];
    _p = p;
  }

  inline T& operator[](int i) {
    check_range(i, _dim, 1);
    return _p[i + _accum];
  }

  inline const T& operator[](int i) const {
    check_range(i, _dim, 1);
    return _p[i + _accum];
  }
};

template <typename T, std::size_t D>
class tensor {
 private:
  T* _data;                         // a pointer to the data
  std::size_t _num_elements;        // total elements of tensor
  std::vector<int> _dims;           // the num of elements in each dimension
  std::vector<int> _strides;        // access strides
  tensor_extent<T, D - 1> _extents; // inner struct to calculate index

 public:
  /*--- typedefs ---*/
  typedef T type;
  typedef std::array<int, D> multi_index;

  template <std::size_t _D>
  using view = tensor_view<T, _D>;

  template <std::size_t _D>
  using fixed_indices = std::array<int, _D>;

  /*--- constructors ---*/
  tensor()
  : _data(nullptr)
  , _dims(D)
  , _strides(D) {}

  // constructor: acceptes initililzer_list whose size is D
  tensor(std::initializer_list<int> i_list)
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

  tensor(const tensor_view<T, D>& view)
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

  tensor_extent<T, D - 1>& operator[](int i) {
    check_range(i, _dims[D - 1], D);
    return _extents.calc_index(i * _strides[D - 1]);
  }

  const tensor_extent<T, D - 1>& operator[](int i) const {
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

  int num_elements() const { return _num_elements; }

  void reshape(const std::array<int, D>& shapes) {
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
    _extents.init(_data, _strides);
  }

  int shape() const { return D; }

  int shape(const int d) const { return _dims[d - 1]; }

  const std::vector<int>& dims() const { return _dims; }

  int strides(const int d) const { return _strides[d]; }

  const std::vector<int>& strides() const { return _strides; }

  T& with_indices(const multi_index& indices) {
    for (int i = D - 1; i >= 0; --i)
      check_range(indices[D - 1 - i], _dims[i], i + 1);
    int idx = 0;
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

  template <std::size_t _D>
  view<_D> make_view(const fixed_indices<D - _D>& indices) {
    static_assert(D - _D > 0, "dimension of view must be greater than 0.");
    multi_index midx;
    std::fill(midx.begin(), midx.end(), 0);
    std::copy(indices.begin(), indices.end(), midx.begin());
    return view<_D>(&with_indices(midx), _dims, _strides);
  }

  ~tensor() { delete[] _data; }
};

template <typename T>
class tensor<T, 1> {
 private:
  T* _data;                  // a pointer to the data
  std::size_t _num_elements; // total elements of tensor
  std::vector<int> _dims;    // the num of elements in each dimension
  std::vector<int> _strides; // access strides

 public:
  /*--- typedefs ---*/
  typedef T type;
  typedef int multi_index;

  // default constructor
  tensor()
  : _data(nullptr)
  , _dims(1)
  , _strides(1) {}

  // constructor: acceptes initililzer_list whose size is D
  tensor(std::size_t d)
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

  tensor(const tensor_view<T, 1>& view)
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

  T& operator[](int i) {
    check_range(i, _dims[0], 1);
    return _data[i];
  }

  const T operator[](int i) const {
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

  int num_elements() const { return _num_elements; }

  int shape() const { return 1; }

  int shape(const int d) const { return _dims[d - 1]; }

  const std::vector<int>& dims() const { return _dims; }

  int strides(const int d) const { return _strides[d]; }

  const std::vector<int>& strides() const { return _strides; }

  T& with_indices(const int indices) {
    check_range(indices, _dims[0], 1);
    return _data[indices];
  }

  ~tensor() { delete[] _data; }
};

/*--- typedefs ---*/
template <typename T>
using vector = tensor<T, 1>;

} // namespace rnz