#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <vector>

namespace rnz {

template <typename T, std::size_t D> struct tensor_extent;
template <typename T, std::size_t D> class tensor;

template <typename T, std::size_t D> class tensor_view {
private:
  T *_data; // a pointer to the first data of dimension D
  int _N;
  std::vector<int> _dims;    // the num of elements in each dimension
  std::vector<int> _strides; // access strides
  tensor_extent<T, D - 1> _extents;

public:
  typedef std::array<int, D> multi_index;
  template <std::size_t _D> using view = tensor_view<T, _D>;

  tensor_view() : _data(nullptr), _dims(D), _strides(D) {}
  tensor_view(T *p, const std::vector<int> &dims,
              const std::vector<int> &strides)
      : _data(p), _dims(D), _strides(D) {
    const int size_diff = dims.size() - D;

    std::copy(std::begin(dims), std::end(dims) - size_diff, std::begin(_dims));
    std::copy(std::begin(strides), std::end(strides) - size_diff,
              std::begin(_strides));

    _N = std::accumulate(std::begin(_dims), std::end(_dims), 1,
                         std::multiplies<int>());

    _extents.init(_data, _strides);
  }

  tensor_view(const tensor_view &src) : _data(src.data()), _N(src.N()) {
    std::cout << "copy ctor" << std::endl;

    _dims = src.dims();
    _strides = src.strides();

    _extents.init(_data, _strides);
  }

  tensor_view(tensor_view &&src) : _data(src.data()), _N(src.N()) {
    std::cout << "move ctor" << std::endl;
    _dims = std::move(src.dims());
    _strides = std::move(src.strides());

    _extents.init(_data, _strides);
  }

  tensor_view &operator=(const tensor_view &src) {
    std::cout << "copy =" << std::endl;
    _data = src.data();
    _N = src.N();
    _dims = src.dims();
    _strides = src.strides();

    _extents.init(_data, _strides);

    return *this;
  }

  tensor_view &operator=(tensor_view &&src) {
    std::cout << "move =" << std::endl;
    _data = src.data();
    _N = src.N();
    _dims = std::move(src.dims());
    _strides = std::move(src.strides());

    _extents.init(_data, _strides);
    return *this;
  }

  tensor_extent<T, D - 1> &operator[](int i) {
    return _extents.calc_index(i * _strides[D - 1]);
  }

  const tensor_extent<T, D - 1> &operator[](int i) const {
    return _extents.calc_index(i * _strides[D - 1]);
  }

  T *begin() { return _data; }
  T *end() { return _data + _N; }

  const T *const begin() const { return _data; }
  const T *const end() const { return _data + _N; }

  T *&data() { return _data; }
  T *const &data() const { return _data; }

  int N() const { return _N; }

  int shape() const { return D; }
  int shape(const int d) const { return _dims[d - 1]; }
  const std::vector<int> &dims() const { return _dims; }

  int strides(const int d) const { return _strides[d]; }
  const std::vector<int> &strides() const { return _strides; }

  T &with_indices(const multi_index &indices) {
    // i* strides(3) + j* strides(2) * k* strides(1) + l;
    int idx = 0;
    for (int i = D - 1; i > 0; --i)
      idx += indices[D - 1 - i] * strides(i);
    idx += indices[D - 1];
    return _data[idx];
  }

  template <typename... Args> T &with_indices(Args... args) {
    return with_indices({args...});
  }

  tensor<T, D> to_tensor() { return tensor<T, D>(*this); }

  template <std::size_t _D>
  view<_D> make_view(const std::array<int, D - _D> &indices) {
    multi_index midx;
    std::fill(midx.begin(), midx.end(), 0);
    std::copy(indices.begin(), indices.end(), midx.begin());
    // std::reverse(midx.begin(), midx.end());
    return view<_D>(&with_indices(midx), _dims, _strides);
  }
};

template <typename T> class tensor_view<T, 1> {
private:
  T *_data;                  // a pointer to the data
  std::size_t _N;            // total elements of tensor
  std::vector<int> _dims;    // the num of elements in each dimension
  std::vector<int> _strides; // access strides
  // tensor_extent<T, D - 1> _extents;  // inner struct to calculate index

public:
  typedef T type;
  typedef int multi_index;

  tensor_view() : _data(nullptr), _N(0), _dims(1), _strides(1) {}

  // constructor: acceptes initililzer_list whose size is D
  tensor_view(T *p, const std::vector<int> &dims,
              const std::vector<int> &strides)
      : _data(p), _dims(1), _strides(1) {
    // std::reverse(_dims.begin(), _dims.end());

    _N = dims[0];
    _dims[0] = _N;
    _strides[0] = 1;
  }

  tensor_view(const tensor_view &src) : _data(src.data()), _N(src.N()) {
    _dims = src.dims();
    _strides = src.strides();
  }

  tensor_view(tensor_view &&src) : _data(src.data()), _N(src.N()) {
    _dims = std::move(src.dims());
    _strides = std::move(src.strides());
  }

  tensor_view &operator=(const tensor_view &src) {
    _data = src.data();
    _N = src.N();
    _dims = src.dims();
    _strides = src.strides();

    return *this;
  }

  tensor_view &operator=(tensor_view &&src) {
    _data = src.data();
    _N = src.N();
    _dims = std::move(src.dims());
    _strides = std::move(src.strides());

    return *this;
  }

  T &operator[](int i) { return _data[i]; }

  const T operator[](int i) const { return _data[i]; }

  T *begin() { return _data; }
  T *end() { return _data + _N; }

  const T *const begin() const { return _data; }
  const T *const end() const { return _data + _N; }

  T *&data() { return _data; }
  T *const data() const { return _data; }

  void fill(T x) { std::fill(_data, _data + _N, x); }

  int N() const { return _N; }

  int shape() const { return 1; }
  int shape(const int d) const { return _dims[d - 1]; }
  const std::vector<int> &dims() const { return _dims; }

  int strides(const int d) const { return _strides[d]; }
  const std::vector<int> &strides() const { return _strides; }

  T &with_indices(const int indices) { return _data[indices]; }

  tensor<T, 1> to_tensor() { return tensor<T, 1>(*this); }
};

template <typename T, std::size_t D> struct tensor_extent {
  tensor_extent<T, D - 1> _extents;
  T *_p;
  int _stride; // D-1次元の要素数
  mutable int _accum;

  inline tensor_extent<T, D> &calc_index(int accum) {
    _accum = accum;
    return *this;
  }

  inline const tensor_extent<T, D> &calc_index(int accum) const {
    _accum = accum;
    return *this;
  }

  tensor_extent() : _p(nullptr), _stride(0), _extents(nullptr, 0) {}

  tensor_extent(T *p, int stride)
      : _p(p), _stride(stride), _extents(p, stride) {}

  tensor_extent<T, D - 1> &operator[](int i) {
    return _extents.calc_index(i * _stride + _accum);
  }

  const tensor_extent<T, D - 1> &operator[](int i) const {
    return _extents.calc_index(i * _stride + _accum);
  }

  int accum() const { return _accum; }

  void init(T *p, std::vector<int> &strides) {
    _p = p;
    _stride = strides[D - 1];
    _extents.init(p, strides);
  }
};

template <typename T> struct tensor_extent<T, 1> {
  T *_p;
  mutable int _accum;

  tensor_extent() : _p(nullptr) {}
  tensor_extent(T *p, int stride) : _p(p) {}

  tensor_extent<T, 1> &calc_index(int accum) {
    _accum = accum;
    return *this;
  }

  const tensor_extent<T, 1> &calc_index(int accum) const {
    _accum = accum;
    return *this;
  }

  void init(T *p, std::vector<int> &strides) { _p = p; }

  inline T &operator[](int i) { return _p[i + _accum]; }
  inline const T &operator[](int i) const { return _p[i + _accum]; }
};

template <typename T, std::size_t D> class tensor {
private:
  T *_data;                         // a pointer to the data
  std::size_t _N;                   // total elements of tensor
  std::vector<int> _dims;           // the num of elements in each dimension
  std::vector<int> _strides;        // access strides
  tensor_extent<T, D - 1> _extents; // inner struct to calculate index

public:
  typedef T type;
  typedef std::array<int, D> multi_index;
  template <std::size_t _D> using view = tensor_view<T, _D>;

  // default constructor
  tensor() : _data(nullptr), _dims(D), _strides(D) {}

  // constructor: acceptes initililzer_list whose size is D
  tensor(std::initializer_list<int> i_list)
      : _data(nullptr), _dims(D), _strides(D) {
    // error check
    // if num of argument is not same as D
    if (i_list.size() != D) {
      std::cerr << "error: dimension miss-match" << std::endl;
      std::exit(1);
    }
    std::copy(i_list.begin(), i_list.end(), _dims.begin());
    // if one of arguments is 0
    if (std::find(std::begin(_dims), std::end(_dims), 0) != std::end(_dims)) {
      std::cerr << "error: 0 is not permitted as a size of dimension"
                << std::endl;
      std::exit(1);
    }
    std::reverse(_dims.begin(), _dims.end());

    _N = std::accumulate(_dims.begin(), _dims.end(), 1, std::multiplies<int>());
    _data = new T[_N];

    _strides[D - 1] = _N / _dims[D - 1];
    for (int i = D - 2; i >= 0; i--) {
      _strides[i] = _strides[i + 1] / _dims[i];
    }

    _extents.init(_data, _strides);
  }

  tensor(const tensor_view<T, D> &view)
      : _data(nullptr), _dims(D), _strides(D) {
    _N = view.N();
    _dims = view.dims();
    _strides = view.strides();

    _data = new T[_N];
    std::copy(view.begin(), view.end(), _data);

    _extents.init(_data, _strides);
  }

  tensor(const tensor &src) {
    _N = src.N();
    _data = new T[_N];
    std::copy(src.begin(), src.end(), _data);

    _dims = src.dims();
    _strides = src.strides();

    _extents.init(_data, _strides);
  }

  tensor(tensor &&src) {
    _N = src.N();
    _data = src.data();
    src.data() = nullptr;
    // std::copy(src.begin(), src.end(), _data);

    _dims = std::move(src.dims());
    _strides = std::move(src.strides());

    _extents.init(_data, _strides);
  }

  tensor &operator=(const tensor &src) {
    delete[] _data;

    _N = src.N();
    _data = new T[_N];
    std::copy(src.begin(), src.end(), _data);

    _dims = src.dims();
    _strides = src.strides();

    _extents.init(_data, _strides);

    return *this;
  }

  tensor &operator=(tensor &&src) {
    delete[] _data;

    _N = src.N();
    _data = src.data();
    // std::copy(src.begin(), src.end(), _data);

    _dims = std::move(src.dims());
    _strides = std::move(src.strides());

    _extents.init(_data, _strides);

    return *this;
  }

  tensor_extent<T, D - 1> &operator[](int i) {
    return _extents.calc_index(i * _strides[D - 1]);
  }

  const tensor_extent<T, D - 1> &operator[](int i) const {
    return _extents.calc_index(i * _strides[D - 1]);
  }

  T *begin() { return _data; }
  T *end() { return _data + _N; }

  const T *const begin() const { return _data; }
  const T *const end() const { return _data + _N; }

  T *&data() { return _data; }
  const T *const data() const { return _data; }

  void fill(T x) { std::fill(_data, _data + _N, x); }

  int N() const { return _N; }

  void reshape(const std::array<int, D> &shapes) {
    if (std::find(std::begin(shapes), std::end(shapes), 0) !=
        std::end(shapes)) {
      std::cerr << "error: 0 is not permitted as a size of dimension"
                << std::endl;
      std::exit(1);
    }

    _N = std::accumulate(std::begin(shapes), std::end(shapes), 1,
                         std::multiplies<int>());
    // for (int i = 0; i < D; i++) _dims[i] = shapes[i];

    std::copy(std::begin(shapes), std::end(shapes), std::begin(_dims));
    std::reverse(std::begin(_dims), std::end(_dims));

    _strides[D - 1] = _N / _dims[D - 1];
    for (int i = D - 2; i >= 0; i--) {
      _strides[i] = _strides[i + 1] / _dims[i];
    }

    delete[] _data;
    _data = new T[_N];
    _extents.init(_data, _strides);
  }

  int shape() const { return D; }
  int shape(const int d) const { return _dims[d - 1]; }
  const std::vector<int> &dims() const { return _dims; }

  int strides(const int d) const { return _strides[d]; }
  const std::vector<int> &strides() const { return _strides; }

  T &with_indices(const multi_index &indices) {
    // i* strides(3) + j* strides(2) * k* strides(1) + l;
    int idx = 0;
    for (int i = D - 1; i > 0; --i)
      idx += indices[D - 1 - i] * strides(i);
    idx += indices[D - 1];
    return _data[idx];
  }

  template <typename... Args> T &with_indices(Args... args) {
    return with_indices({args...});
  }

  template <std::size_t _D>
  view<_D> make_view(const std::array<int, D - _D> &indices) {
    multi_index midx;
    std::fill(midx.begin(), midx.end(), 0);
    std::copy(indices.begin(), indices.end(), midx.begin());
    // std::reverse(midx.begin(), midx.end());
    return view<_D>(&with_indices(midx), _dims, _strides);
  }

  ~tensor() { delete[] _data; }
};

template <typename T> class tensor<T, 1> {
private:
  T *_data;                  // a pointer to the data
  std::size_t _N;            // total elements of tensor
  std::vector<int> _dims;    // the num of elements in each dimension
  std::vector<int> _strides; // access strides
  // tensor_extent<T, D - 1> _extents;  // inner struct to calculate index

public:
  typedef T type;
  typedef int multi_index;

  // default constructor
  tensor() : _data(nullptr), _dims(1), _strides(1) {}

  // constructor: acceptes initililzer_list whose size is D
  tensor(std::size_t d) : _data(nullptr), _dims(1), _strides(1) {
    // error check
    // if one of arguments is 0
    if (d == 0) {
      std::cerr << "error: 0 is not permitted as a size of dimension"
                << std::endl;
      std::exit(1);
    }
    // std::reverse(_dims.begin(), _dims.end());

    _N = d;
    _data = new T[_N];
    _dims[0] = _N;
    _strides[0] = 1;
  }

  tensor(const tensor &src) {
    _N = src.N();
    _data = new T[_N];
    std::copy(src.begin(), src.end(), _data);

    _dims = src.dims();
    _strides = src.strides();
  }

  tensor(const tensor_view<T, 1> &view)
      : _data(nullptr), _dims(1), _strides(1) {
    _N = view.N();
    _dims = view.dims();
    _strides = view.strides();

    _data = new T[_N];
    std::copy(std::begin(view), std::end(view), _data);
  }

  tensor(tensor &&src) {
    _N = src.N();
    _data = src.data();
    src.data() = nullptr;
    // std::copy(src.begin(), src.end(), _data);

    _dims = std::move(src.dims());
    _strides = std::move(src.strides());
  }

  tensor &operator=(const tensor &src) {
    delete[] _data;

    _N = src.N();
    _data = new T[_N];
    std::copy(src.begin(), src.end(), _data);

    _dims = src.dims();
    _strides = src.strides();

    return *this;
  }

  tensor &operator=(tensor &&src) {
    delete[] _data;

    _N = src.N();
    _data = src.data();
    src.data() = nullptr;
    // std::copy(src.begin(), src.end(), _data);

    _dims = std::move(src.dims());
    _strides = std::move(src.strides());

    return *this;
  }

  T &operator[](int i) { return _data[i]; }

  const T operator[](int i) const { return _data[i]; }

  T *begin() { return _data; }
  T *end() { return _data + _N; }

  const T *const begin() const { return _data; }
  const T *const end() const { return _data + _N; }

  T *&data() { return _data; }
  const T *const data() const { return _data; }

  void fill(T x) { std::fill(_data, _data + _N, x); }

  int N() const { return _N; }

  int shape() const { return 1; }
  int shape(const int d) const { return _dims[d - 1]; }
  const std::vector<int> &dims() const { return _dims; }

  int strides(const int d) const { return _strides[d]; }
  const std::vector<int> &strides() const { return _strides; }

  T &with_indices(const int indices) { return _data[indices]; }

  ~tensor() { delete[] _data; }
};

template <typename T> using vector = tensor<T, 1>;

} // namespace rnz