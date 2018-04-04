#pragma once
#include <iterator>
#include <tuple>
#include <utility>
#include <memory>
#include <optional>

//------------------------------- BEGIN: helper functions -------------------------------

namespace detail
{
	template<typename... Ts, std::size_t... Idx>
	void inc_all(std::tuple<Ts...>& t, std::index_sequence<Idx...>)
	{
		(..., ++std::get<Idx>(t));
	}

	template<typename... Ts, typename... Us, std::size_t... Idx>
	bool or_all(const std::tuple<Ts...>& t1, const std::tuple<Us...>& t2, std::index_sequence<Idx...>)
	{
		return (... || (std::get<Idx>(t1) == std::get<Idx>(t2)));
	}

	template<typename... Ts, std::size_t... Idx>
	void emplace_deref(std::optional<std::tuple<decltype(*std::declval<Ts>())...>>& opt, std::tuple<Ts...>& t, std::index_sequence<Idx...>)
	{
		opt.emplace(*std::get<Idx>(t)...);
	}

	template<typename... Ts, typename FuncT, std::size_t... Idx>
	auto extract(std::tuple<Ts...>& t, FuncT f, std::index_sequence<Idx...>)
	{
		return std::tuple<decltype(f(std::get<Idx>(t)))...>{ f(std::get<Idx>(t))... };
	}
}

//------------------------------- END: helper functions ---------------------------------

//------------------------------- BEGIN: zip_iter -------------------------------

template<typename... Ts>
class zip_iter
{
	using extracted_type = std::tuple<decltype(*std::declval<Ts>())...>;

	std::tuple<Ts...> iters_;
	std::optional<extracted_type> ele_;
public:	
	using difference_type	= std::ptrdiff_t			;
	using value_type		= extracted_type			;
	using pointer			= extracted_type *			;
	using reference			= extracted_type &			;
	using iterator_category = std::forward_iterator_tag	;

	zip_iter() = default;
	zip_iter(std::tuple<Ts...> iters) : iters_(std::move(iters)) {}

	const auto& iters() const { return iters_; }

	zip_iter& operator++()
	{
		detail::inc_all(iters_, std::index_sequence_for<Ts...>{});
		ele_.reset();
		return *this;
	}

	zip_iter operator++(int)
	{
		zip_iter temp(*this);
		++*this;
		return temp;
	}

	bool operator==(const zip_iter& other) const { return detail::or_all(iters_, other.iters_, std::index_sequence_for<Ts...>{}); }
	bool operator!=(const zip_iter& other) const { return !(*this == other); }

	reference operator*()
	{
		if (!ele_)
			detail::emplace_deref(ele_, iters_, std::index_sequence_for<Ts...>{}); //optional value must be emplace constructed if extracted_type contains const items
		return ele_.value();
	}

	pointer operator->() { return std::addressof(**this); }
};

//------------------------------- END: zip_iter ---------------------------------

//------------------------------- BEGIN: zip_sentienel -------------------------------

template<typename... Ts>
class zip_sentinel
{
	std::tuple<Ts...> sentinels_;
public:
	zip_sentinel(std::tuple<Ts...> sentinels) : sentinels_(std::move(sentinels)) {}
	const auto& sentinels() const { return sentinels_; }
};

template<typename... Ts, typename... Us>
bool operator==(const zip_iter<Ts...>& it, const zip_sentinel<Us...>& sen)
{
	return detail::or_all(it.iters(), sen.sentinels(), std::index_sequence_for<Ts...>{});
}

template<typename... Ts, typename... Us>
bool operator==(const zip_sentinel<Ts...>& sen, const zip_iter<Us...>& it)
{
	return detail::or_all(it.iters(), sen.sentinels(), std::index_sequence_for<Ts...>{});
}

template<typename... Ts, typename... Us> bool operator!=(const zip_iter<Ts...>& it,      const zip_sentinel<Us...>& sen) { return !(it == sen); }
template<typename... Ts, typename... Us> bool operator!=(const zip_sentinel<Ts...>& sen, const zip_iter<Us...>& it)      { return !(it == sen); }

//------------------------------- END: zip_sentinel ---------------------------------

//------------------------------- BEGIN: zip_set -------------------------------

//valid as long as all lvalues passed to constructor are valid
//begin == end for zip iterator as soon as begin == end for any of its held tuple of iterators
template<typename... Ts>
class zip_set
{
	std::tuple<Ts...> iterables_;
public:
	zip_set(Ts&&... iterables) : iterables_{ std::forward<Ts>(iterables)... } {}

	zip_iter<decltype(std::declval<Ts>().begin())...> begin()
	{
		return { detail::extract(iterables_, [](auto& iterable) { return iterable.begin(); }, std::index_sequence_for<Ts...>{}) };
	}

	zip_sentinel<decltype(std::declval<Ts>().end())...> end()
	{
		return { detail::extract(iterables_, [](auto& iterable) { return iterable.end(); }, std::index_sequence_for<Ts...>{}) };
	}
};

template<typename... Ts>
zip_set<Ts...> zip(Ts&&... iterables) { return { std::forward<Ts>(iterables)... }; }

//------------------------------- END: zip_set ---------------------------------
