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

template <typename T, std::size_t D> struct tensor_extent {
  tensor_extent<T, D - 1> _extents;
  T *_p;
  int _stride; // D-1次元の要素数
  int _accum;

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
  int _accum;

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