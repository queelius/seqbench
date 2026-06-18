#include "test_util.hpp"
#include <torch/torch.h>
#include "seqbench/corpus.hpp"

static void test_torch_works() {
  torch::manual_seed(0);
  auto a = torch::tensor({1.0, 2.0, 3.0});
  CHECK(a.sum().item<double>() == 6.0);
  auto m = torch::ones({2, 3});
  CHECK(m.size(0) == 2);
  CHECK(m.size(1) == 3);
}

static void test_bench_links() {
  auto toy = seqbench::toy_corpus();
  CHECK(toy.len > 0);
}

int main() {
  RUN(test_torch_works);
  RUN(test_bench_links);
  return test_summary();
}
