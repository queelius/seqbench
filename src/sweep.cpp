#include "seqbench/sweep.hpp"
#include "seqbench/metric.hpp"
#include <cstdio>
#include <stdexcept>

namespace seqbench {

std::vector<SweepPoint> run_sweep(
    const std::function<std::unique_ptr<Model>(int)>& factory,
    const std::vector<int>& knobs, ByteSpan data) {
  std::vector<SweepPoint> pts;
  pts.reserve(knobs.size());
  for (int k : knobs) {
    std::unique_ptr<Model> m = factory(k);
    BpbResult r = run_adaptive(*m, data);
    pts.push_back(SweepPoint{k, r.bpb(), r.seconds});
  }
  return pts;
}

void write_csv(const std::vector<SweepPoint>& pts, const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "w");
  if (!f) throw std::runtime_error("write_csv: cannot open " + path);
  std::fprintf(f, "knob,bpb,seconds\n");
  for (const SweepPoint& p : pts)
    std::fprintf(f, "%d,%.6f,%.6f\n", p.knob, p.bpb, p.seconds);
  std::fclose(f);
}

}  // namespace seqbench
