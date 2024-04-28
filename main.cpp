#include <iostream>
#include "avx2_prefix.hpp"
#include <cstdint>
#include <numeric>


int main() {
    std::size_t size = 32;
    std::vector<uint8_t> data(size);

    std::iota(data.begin(), data.end(),0);
    std::vector data_copy(data);
    std::inclusive_scan(data_copy.begin(), data_copy.end(), data_copy.begin(), std::plus<>());
    inplace_prefix_sum(data);
    std::cout << (data == data_copy) << std::endl;
    return 0;

}
