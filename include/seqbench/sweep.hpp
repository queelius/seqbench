#pragma once
#include "seqbench/byte_span.hpp"
#include "seqbench/model.hpp"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace seqbench {

struct SweepPoint {
  int knob = 0;
  double bpb = 0.0;
  double seconds = 0.0;  // wall-clock compute proxy for v1
};

std::vector<SweepPoint> run_sweep(
    const std::function<std::unique_ptr<Model>(int)>& factory,
    const std::vector<int>& knobs, ByteSpan data);

void write_csv(const std::vector<SweepPoint>& pts, const std::string& path);

}  // namespace seqbench
