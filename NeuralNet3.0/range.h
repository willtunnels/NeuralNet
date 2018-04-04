#pragma once
#include <iterator>
#include <memory>

template<typename T>
class range_iter
{
	T idx_, step_;
public:
	using difference_type	= std::ptrdiff_t			;
	using value_type		= T							;
	using pointer			= const T *					;
	using reference			= const T &					;
	using iterator_category = std::forward_iterator_tag	;

	range_iter() = default;
	range_iter(T idx, T step) : idx_{ idx }, step_{ step } {}

	range_iter& operator++()
	{
		idx_ += step_;
		return *this;
	}

	range_iter operator++(int)
	{
		range_iter temp(*this);
		++*this;
		return temp;
	}

	bool operator==(range_iter other) const { return idx_ == other.idx_; }
	bool operator!=(range_iter other) const { return !(*this == other); }

	reference operator*()  const { return idx_; }
	pointer   operator->() const { return std::addressof(idx_); }
};

template<typename T>
class range_sentinel
{
	T end_;
public:
	range_sentinel(T end) : end_{ end } {}
	T end() const { return end_; }
};

template<typename T> bool operator==(range_iter<T> it,      range_sentinel<T> sen) { return *it >= sen.end(); }
template<typename T> bool operator==(range_sentinel<T> sen, range_iter<T> it)      { return *it >= sen.end(); }

template<typename T> bool operator!=(range_iter<T> it,      range_sentinel<T> sen) { return !(it == sen); }
template<typename T> bool operator!=(range_sentinel<T> sen, range_iter<T> it)      { return !(it == sen); }

template<typename T>
class range
{
	T begin_, end_, step_;
public:
	range(T begin, T end, T step = 1) : begin_{ begin }, end_{ end }, step_{ step } {}
	range_iter<T> begin() { return { begin_, step_ }; }
	range_sentinel<T> end() { return { end_ }; }
};

using crange = range<char>;
using irange = range<int>;
using frange = range<float>;
using drange = range<double>;
