#include "test_util.hpp"
#include "seqbench/sweep.hpp"
#include "seqbench/context_model.hpp"
#include "seqbench/corpus.hpp"
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace seqbench;

static void test_sweep_endpoints_improve_on_toy() {
  std::vector<int> knobs = {0, 1, 2, 3};
  std::vector<SweepPoint> pts =
      run_sweep(make_context_model, knobs, toy_corpus());
  CHECK(pts.size() == 4);
  CHECK(pts[0].knob == 0);
  CHECK(pts[3].knob == 3);
  // bpb improves (drops) as order rises on the repetitive toy corpus.
  CHECK(pts[3].bpb < pts[0].bpb);
}

static void test_write_csv() {
  std::vector<SweepPoint> pts = {{0, 4.5, 0.01}, {1, 3.0, 0.02}};
  const std::string path = "/tmp/seqbench_sweep_test.csv";
  write_csv(pts, path);
  std::ifstream f(path);
  std::string header;
  std::getline(f, header);
  CHECK(header == "knob,bpb,seconds");
  std::string line;
  std::getline(f, line);
  CHECK(line.rfind("0,", 0) == 0);
  f.close();
  std::remove(path.c_str());
}

int main() {
  RUN(test_sweep_endpoints_improve_on_toy);
  RUN(test_write_csv);
  return test_summary();
}
